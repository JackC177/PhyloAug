import numpy as np
import random
import json
from collections import Counter
import os
from src.parse_rna_fam_data import remote_blastn_find_homologs_subseq_only
from src.run_paml_non_coding_file import run_paml
from src.get_phylogenetic_tree_from_msa_fasttree import get_phylogenetic_trees_fasttree
from collections import Counter
import glob
import numpy as np
from collections import Counter
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from collections import Counter
import random
from pathlib import Path


def read_stockholm_and_find_conserved(file_path, num_sequences=20, threshold=95):
    alignment = AlignIO.read(file_path, "stockholm")
    print(f"Total sequences in alignment: {len(alignment)}")
    
    selected_seqs = random.sample(list(alignment), min(num_sequences, len(alignment)))
    print(f"Selected {len(selected_seqs)} sequences.")
    
    aln_len = alignment.get_alignment_length()
    
    # Compute conservation per column
    conserved_positions = []
    for i in range(aln_len):
        column = [record.seq[i] for record in alignment]
        freqs = Counter(column)
        if '-' in freqs:
            continue  # ignore gaps
        if not freqs:
            continue
        base, freq = freqs.most_common(1)[0]
        conservation_ratio = freq / len(alignment)
        if conservation_ratio >= threshold:
            conserved_positions.append((i, base, conservation_ratio))
    
    print("\nConserved Positions (index, base, conservation %):")
    for idx, base, cons in conserved_positions:
        print(f"{idx:4d}  {base}  {cons:.2f}")
        
    selected_seqs = random.sample(list(alignment), min(num_sequences, len(alignment)))
    
    return conserved_positions, selected_seqs


# Gather the data
task = "test_rfam_data"
seq_data = []
conserved_nucs = []
og_seqs = []
for file in os.listdir("."):
    if file[-13:] == "stockholm.txt":
        conserved, seqs = read_stockholm_and_find_conserved(file, num_sequences=50, threshold=0.70)
        rfam_fam_sequences = [
            (i, SeqRecord(Seq(str(seq.seq).replace("-", "")), id=seq.id, description=""))
            for i, seq in enumerate(seqs)
        ]
        seq_data.append([file, rfam_fam_sequences])
        conserved_nucs.append(conserved)
        og_seqs.append(seqs[0])

# Run nt database to collect homologs
# Check to see if BLASTN has been run already (xml file)
os.makedirs("rfam_test_data_train_families", exist_ok=True)
# Example RNA sequences
for file, seqs in seq_data:
    seq_records = [
        SeqRecord(Seq(seq.seq), id="seq"+str(seq_id), description="")
        for seq_id, seq in seqs
    ]
    count = SeqIO.write(seq_records, "rfam_test_data_train_families/"+file[:7]+".fasta", "fasta")
    print(f"Wrote {count} records to {file[:7]}.fasta")

task_folder_name = "rfam_test_data"
rnas = []
with open(task_folder_name+"/train.json", "r") as f:
    for i, line in enumerate(f.readlines()):
        rnas.append(json.loads(line.strip())["seq"])
remote_blastn_find_homologs_subseq_only(rnas, output_xml=f"{task_folder_name}/blastn_{task}.xml", output_dir=f"{task_folder_name.replace('/','')}_train_families/")
    
# Get Phylogenetic Trees for PAML Analysis
tree_files = glob.glob(os.path.join(f"{task_folder_name.replace('/','')}_train_families", "*.tree"))
if not tree_files:
    get_phylogenetic_trees_fasttree(f"{task_folder_name.replace('/','')}_train_families")
# Perform PAML Analysis to identify neutral mutation areas and mask these areas
masked_data = run_paml(Path(f"{task_folder_name.replace('/','')}_train_families"), output_json_path=f"{task_folder_name}/masked_train.json")

masked_seqs = [data["masked_sequence"].replace("<mask>","X").replace("T","U") for data in masked_data]

for seq_idx, (seq, pos_list) in enumerate(zip(masked_seqs, conserved_nucs)):
    print(f"{len(pos_list)} conserved nucleotides")
    print(f"stockholm alignment length: {len(og_seqs[seq_idx])}")
    print(f"masked sequence length: {len(seq)}")
    seq_report = []
    failed = 0
    for pos in pos_list:
        if pos[0] < len(seq):
            base = seq[pos[0]]
            is_masked = base == "X"
        else:
            base = None
            is_masked = False  # out-of-bounds treated as not masked
        if is_masked:
            failed += 1
        seq_report.append((pos[0], base, is_masked))
    print(f"{failed} number of masked nucleotides on conserved positions")
