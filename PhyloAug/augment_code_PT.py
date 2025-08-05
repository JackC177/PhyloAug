import numpy as np
import torch
import random
import json
import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import autocuda
from torch.cuda.amp import autocast
from accelerate import Accelerator
from collections import Counter
import re
import RNA
import os
from src.parse_rna_fam_data import remove_non_rfam_families, json_to_fasta, remote_blastn_find_homologs_subseq_only
from src.create_fasta_from_df import get_structure_lookup_dict, add_secondary_structure_to_json, add_label_to_json
from src.run_paml_non_coding_file import run_paml
from src.get_phylogenetic_tree_from_msa_fasttree import get_phylogenetic_trees_fasttree
from collections import Counter
import glob
import numpy as np
from scipy.spatial.distance import jensenshannon
from collections import Counter
from pathlib import Path


def normalize_distribution(counter, alphabet="ACGT"):
    total = sum(counter.get(base, 0) for base in alphabet)
    if total == 0:
        # If no valid bases, return a uniform or zero distribution
        return np.zeros(len(alphabet))
    return np.array([counter.get(base, 0) / total for base in alphabet])

def jsd_between_distributions(dist1, dist2):
    return jensenshannon(dist1, dist2, base=2)

def compute_column_distribution(sequences, position, alphabet="ACGT"):
    col = [seq[position] for seq in sequences if seq[position] in alphabet]
    return Counter(col)


class OmniGenomeModelForAugmentation(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path=None,
        max_length=2052,
        instance_num=1,
        prob_cutoff=0.10,
        top_k=3,
        structure_aware=True,
        *args,
        **kwargs
    ):
        """
        Initialize the model, tokenizer, and augmentation hyperparameters.

        Parameters:
        - model_name_or_path (str): Path or model name for loading the pre-trained model.
        - noise_ratio (float): The proportion of tokens to mask in each sequence for augmentation.
        - max_length (int): The maximum sequence length for tokenization.
        - instance_num (int): Number of augmented instances to generate per sequence.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.model = self.accelerator.prepare(self.model)
        self.device = autocuda.auto_cuda()
        self.device = self.accelerator.device
        self.model.to(self.device)

        # Hyperparameters for augmentation
        self.max_length = max_length
        self.k = instance_num
        self.prob_cutoff = prob_cutoff
        self.top_k = top_k
        self.structure_aware = structure_aware

    def load_sequences_from_file(self, input_file, structures_available=True, mRNA=False):
        """Load sequences from a JSON file."""
        masked_sequences = []
        original_sequences = []
        structures = []
        labels = []
        # Ensure all tokens are valid
        allowed_mask_tokens = ["A", "C", "G", "T", "U", "a", "c", "g", "t", "u", "<mask>"]
        allowed_tokens = ["A", "C", "G", "T", "U", "a", "c", "g", "t", "u"]
        if mRNA:
            target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        else:
            target_cols = ["label"]
        with open(input_file, "r") as f:
            for i, line in enumerate(f.readlines()):
                masked_seq = self.tokenize_masked_seq(json.loads(line)["masked_sequence"])
                masked_seq = "".join([x if x in allowed_mask_tokens else "N" for x in masked_seq])
                masked_sequences.append(masked_seq)
                original_seq = json.loads(line)["original_sequence"]
                original_seq = "".join([x for x in original_seq])
                original_sequences.append(original_seq)
                if structures_available:
                    data = json.loads(line)
                    struct = data.get("structure", None)
                else:
                    struct, mfe = RNA.fold(original_seq)
                structures.append(struct)
                temp_labels = []
                for col in target_cols:
                    temp_labels.append(json.loads(line)[col])
                labels.append(temp_labels)
        
        return masked_sequences, structures, original_sequences, labels

    def stochastic_token_select(self, predictions, max_seq_len, prob_cutoff=0, top_k=3):
        """
        Sample tokens from top-k predictions, constrained to allowed tokens.

        Args:
            predictions: Tensor [batch_size, seq_len, vocab_size]
            max_seq_len: Number of masked positions to sample
            prob_cutoff: Probability threshold to ignore low-probability top-k options
            top_k: How many top logits to consider

        Returns:
            sampled_tokens: Tensor [batch_size, seq_len] of token indices
        """
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else self.tokenizer.vocab
        vocab_size = predictions.size(-1)

        # Define allowed and disallowed token IDs
        allowed_chars = ["A", "G", "C", "T", "U", "a", "g", "c", "t", "u"]
        allowed_tokens = {vocab.get(x) for x in allowed_chars}
        disallowed = list(set(range(vocab_size)) - allowed_tokens)

        # Mask disallowed tokens across [1:max_seq_len+1]
        predictions[:, 1:max_seq_len + 1, disallowed] = float('-inf')

        # Convert logits to probabilities and apply top-k
        probs = torch.softmax(predictions, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

        # Normalize top-k probabilities
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        batch_size, seq_len, _ = predictions.shape
        sampled_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)

        # Vectorized sampling for masked RNA region
        count_ns = 0
        for b in range(batch_size):
            for s in range(1, max_seq_len + 1):
                sampled_idx = torch.multinomial(topk_probs[b, s], 1).item()
                token_id = topk_indices[b, s, sampled_idx]

                # Convert "T" to "U", fallback to "N" if invalid
                if token_id == vocab.get("T"):
                    token_id = vocab.get("U")
                elif token_id.item() not in allowed_tokens:
                    token_id = vocab.get("T")

                sampled_tokens[b, s] = token_id

            # For non-RNA positions: argmax (most likely token)
            sampled_tokens[b, max_seq_len + 1:] = torch.argmax(predictions[b, max_seq_len + 1:], dim=-1)
        return sampled_tokens

    def augment_sequence(self, seq, struct):
        """Perform augmentation on a single sequence by predicting masked tokens."""
        # Concatenate sequence and structure for structure-aware augmentation
        if self.structure_aware:
            # Convert pseudoknots to .
            struct = struct.translate(str.maketrans("[]{}<>aAbBcCdDeE", "................"))
            input = f"{seq}<eos>{''.join(struct)}"
            if len(input) > self.max_length:
                print("Sequence and Structure too long, cannot use structure-aware")
                print("Defaulting to non-structure-aware. To fix, increase max_length")
        else:  # Regular augmentation with MSA-based MLM
            input = seq
        tokenized_inputs = self.tokenizer(
            input,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Convert to Half Precision
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                predictions = self.model(**tokenized_inputs.to(self.device))["logits"]
            predicted_tokens = self.stochastic_token_select(predictions.cpu(),
                                                            max_seq_len=len(self.tokenize_masked_seq(seq)))
        
        # Replace masked tokens with predicted tokens
        input_ids = tokenized_inputs["input_ids"][0].cpu()
        mask_token_id = self.tokenizer.mask_token_id
        masked_indices = input_ids == mask_token_id
        
        # Convert predicted token IDs to tokens (strings)
        predicted_token_ids = predicted_tokens[0]
        predicted_tokens_str = self.tokenizer.convert_ids_to_tokens(predicted_token_ids)
        
        # Process replacements
        for idx in torch.where(masked_indices)[0]:
            pred_token = predicted_tokens_str[idx]
            if pred_token == "N":
                replacement = random.choice(["A", "T", "G", "C"])
            else:
                replacement = pred_token
            input_ids[idx] = self.tokenizer.convert_tokens_to_ids(replacement)

        augmented_sequence = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        # Take the structure out (don't do before the decoder in case of a changed tokenizer)
        nucleotide_set = set("CAUTGcautg")
        tester_set = set("cautgCAUTG.() ")
        # assert any(char not in tester_set for char in augmented_sequence)
        new_augmented_sequence = [x for x in augmented_sequence if x in nucleotide_set]
        test_seq = [x for x in augmented_sequence if x in tester_set]
        if len(new_augmented_sequence) != len(self.tokenize_masked_seq(seq)):
            print(new_augmented_sequence)
            print(self.tokenize_masked_seq(seq))
            raise ValueError("Check your inputs, unseen tokens identified")
        #new_augmented_sequence = [
        #    random.choice(["A", "T", "G", "C"]) if x == "N" else x
        #    for x in new_augmented_sequence
        #]

        return "".join(new_augmented_sequence).upper()  # Set lowercase nucs to uppercase

    def tokenize_masked_seq(self, seq):
        # Regex splits into either <mask> or a single character
        tokens = re.findall(r'<mask>|[AGCU]', seq)
        return tokens

    def compute_base_distributions(self, masked_seq, sequences):
        """Compute position-wise base frequency distributions."""
        # Get masked positions
        masked_seq = self.tokenize_masked_seq(masked_seq)
        seq_positions = [i for i, seq_char in enumerate(masked_seq) if seq_char == "<mask>"]
        distributions = []

        for pos in seq_positions:
            counter = Counter(seq[pos] for seq in sequences)
            total = sum(counter.values())
            # Order: A, C, G, U
            freqs = [
                counter.get('A', 0) / total,
                counter.get('C', 0) / total,
                counter.get('G', 0) / total,
                counter.get('U', 0) / total
            ]
            distributions.append(freqs)

        return np.array(distributions)

    def augment(self, seq, struct, k=10):
        """Generate multiple augmented instances for a single sequence."""
        augmented_sequences = []
        matching_structs = []
        i = 0
        while i < k:
            augmented_seq = self.augment_sequence(seq, struct).strip().replace(" ","")  # Remove whitespace
            augmented_sequences.append(augmented_seq)
            if len(augmented_sequences) > 1:
                if i < self.aug_attempts:
                    augmented_sequences.pop(-1)
                    break
            matching_structs.append(struct)
            i += 1
        return augmented_sequences, matching_structs

    def augment_sequences(self, sequences, structs, labels):
        """Augment a list of sequences by applying noise and performing MLM-based predictions."""
        all_augmented_sequences = []
        num_skipped = 0
        if len(labels) != len(sequences):
            labels = structs
        for i, temp in tqdm.tqdm(enumerate(zip(sequences, structs, labels)), desc="Augmenting Sequences", total=len(sequences)):
            seq = temp[0]
            struct = temp[1]
            label = temp[2]
            # Ensure there are sufficient positions to mask in the sequence
            split_seq = self.tokenize_masked_seq(seq)
            masked_tokens = [1 for char in split_seq if char == "<mask>"]
            if sum(masked_tokens) < 1:
                continue
            augmented_instances, struct_label = self.augment(seq, struct, self.k)
            for augmented_instance, struct_l in zip(augmented_instances, struct_label):
                all_augmented_sequences.append([augmented_instance, label]) # Label may not be a structure
        if num_skipped < 1:
            print(f"None skipped!")
        else:
            print(f"Percentage of Sequences Skipped due to distance_cutoff: {num_skipped/len(sequences)}")
        return all_augmented_sequences


    def save_augmented_sequences(self, augmented_sequences, output_file, og_sequences, og_labels, mRNA=False):
        """Save augmented sequences to a JSON file."""
        og_counter = Counter("".join(og_sequences))
        augmented_counter = Counter("".join([x[0] for x in augmented_sequences]))
        with open(output_file, "w") as f:
            for data in augmented_sequences:
                # print(data)
                if not mRNA:
                    f.write(json.dumps({"seq": "".join(data[0]).replace(" ", "").replace("U", "T"), "label": data[1][0]})+"\n")
                else:
                    f.write(json.dumps(
                        {"seq": "".join(data[0]).replace(" ", "").replace("U", "T"), "reactivity": (data[1][0]),
                         "deg_Mg_pH10": (data[1][1]), "deg_Mg_50C": (data[1][2])})+"\n")

            for og_seq, og_label in zip(og_sequences, og_labels):
                if not mRNA:
                    f.write(json.dumps({"seq": "".join(og_seq).replace(" ", "").replace("U", "T"), "label": (og_label)})+"\n")
                else:
                    f.write(json.dumps({"seq": "".join(og_seq).replace(" ", "").replace("U", "T"), "reactivity": (og_label[0]),
                                        "deg_Mg_pH10": (og_label[1]), "deg_Mg_50C": (og_label[2])})+"\n")

    def extract_n_augments(self, augmented_sequences, n_aug=8):
        assert n_aug <= 12, "You only have 12 augmentations per sequence"
        return [
            augmented_sequences[i + j]
            for i in range(0, len(augmented_sequences), 12)  # every original sequence block
            for j in range(n_aug)  # first `n_aug` in each block
        ]


    def augment_from_file(self, input_file, output_file, structures_available=True, structure_pred=True):
        """Main function to handle the augmentation process from a file input to a file output."""
        sequences, structures, og_sequences, labels = self.load_sequences_from_file(input_file, structures_available=structures_available)
        augmented_sequences = self.augment_sequences(sequences, structures, labels)
        # Process Augmented Sequences Properly
        if structure_pred:  # Structure as label
            self.save_augmented_sequences(augmented_sequences,
                                          output_file+f"{self.k}.json",
                                          og_sequences, structures)
        else:  # Label as Label
            self.save_augmented_sequences(augmented_sequences,
                                          output_file + f"{self.k}.json",
                                          og_sequences, labels, mRNA=False)


if __name__ == "__main__":
    # Generate augmented datasets
    tasks = ["RNA-SSP-Archive2"]
    
    for task in tasks:
        task_folder = [task]
        
        for task_folder_name in task_folder:
            print(task_folder_name)
            # First, turn .json file into .fasta file
            # All training files should be called train.json
            json_to_fasta(task_folder_name+"/train.json", task_folder_name+"/train.fasta")
            
            # Gather rna data from training set
            rnas = []
            with open(task_folder_name+"/train.json", "r") as f:
                for i, line in enumerate(f.readlines()):
                    try:
                        rnas.append(json.loads(line.strip())["seq"])
                    except:
                        rnas.append(json.loads(line.strip())["sequence"])
            # Run nt database to collect homologs
            # Check to see if BLASTN has been run already (xml file)
            remote_blastn_find_homologs_subseq_only(rnas, output_xml=f"{task_folder_name}/blastn_{task}.xml", output_dir=f"{task_folder_name.replace('/','')}_train_families/")
            
            # Remove data not from Rfam families
            remove_non_rfam_families(f"{task_folder_name.replace('/','')}_train_families")

            # Get Phylogenetic Trees for PAML Analysis
            tree_files = glob.glob(os.path.join(f"{task_folder_name.replace('/','')}_train_families", "*.tree"))
            if not tree_files:
                get_phylogenetic_trees_fasttree(f"{task_folder_name.replace('/','')}_train_families")
            
            # Perform PAML Analysis to identify neutral mutation areas and mask these areas
            masked_data = run_paml(Path(f"{task_folder_name.replace('/','')}_train_families"), output_json_path=f"{task_folder_name}/masked_train.json", no_rfam=True)
            
            # Add structure back in (if it exists) ready for augmentation
            if "RNA-SSP" in task:
                structure_lookup = get_structure_lookup_dict(task_folder_name+"/train.json")
                add_secondary_structure_to_json(
                    jsonl_file=task_folder_name+"/masked_train.json",
                    structure_dict=structure_lookup,
                    output_jsonl_file=task_folder_name+"/masked_train_with_structure.json"
                )
                structures_available = True
            else:
                # If structures are not available, keep the labels
                structure_lookup = get_structure_lookup_dict(task_folder_name+"/train.json")
                add_label_to_json(
                    jsonl_file=task_folder_name+"/masked_train.json",
                    og_data_file=task_folder_name+"/train.json",
                    output_jsonl_file=task_folder_name+"/masked_train2.json",
                    structure_dict=structure_lookup,
                    mRNA=False
                )
            structures_available = False

            for aug_num in [1, 2, 4, 8, 12]:
                model = OmniGenomeModelForAugmentation(
                    model_name_or_path="../OmniGenome-186M/",
                    max_length=4096,  # Maximum token length
                    instance_num=aug_num,  # Number of augmented instances per sequence
                )
                
                if structures_available:
                    model.augment_from_file(task_folder_name+"/masked_train_with_structure.json", f"{task_folder_name}/augmented_train_aug_phylogenetic",
                                            structures_available=structures_available)
                else:
                    model.augment_from_file(task_folder_name + "/masked_train2.json",
                                            f"{task_folder_name}/augmented_train_aug_phylogenetic",
                                            structures_available=structures_available,
                                            structure_pred=False)
