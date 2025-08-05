import numpy as np
import argparse
import json
import os
from collections import defaultdict, Counter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import subprocess
import re
import tempfile
from Bio.Blast import NCBIXML
from scipy.spatial.distance import jensenshannon
import random
import math
import RNA
from Bio.SeqIO import FastaIO
from src.GLOBAL_PATHS import *



def get_args():
    parser = argparse.ArgumentParser(description="Process FASTA and Infernal tblout files.")
    parser.add_argument("-f", "--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("-t", "--tblout", required=True, help="Path to Infernal tblout file")
    parser.add_argument("-o", "--output", required=True, help="Path to output JSON file")
    return parser.parse_args()

def is_valid_rna(seq):
    return re.fullmatch(r'[ACGUTNacgutn]+', seq) is not None

def json_to_fasta(json_filename, fasta_filename):
    with open(json_filename, 'r') as json_file, open(fasta_filename, 'w') as fasta_file:
        for i, line in enumerate(json_file, 1):
            try:
                data = json.loads(line)
                if data.get("seq") is None:
                    seq = data.get("sequence")
                else:
                    seq = data.get("seq")
                if not is_valid_rna(seq):
                    print(f"Skipping invalid sequence {seq}")
                    continue
                if seq:
                    fasta_file.write(f">seq{i}\n{seq}\n")
            except json.JSONDecodeError as e:
                print(f"Skipping line {i}: Invalid JSON - {e}")

def parse_tblout(tblout_path):
    hits = defaultdict(list)  # seqid -> list of families (can be extended)
    with open(tblout_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split()
            target_name = fields[0]  # e.g., RFxxxxx
            query_name = fields[2]   # seqid from FASTA
            hits[target_name].append(query_name)
    return hits


def fasta_to_json(fasta_path, tblout_path, output_path):
    hits = parse_tblout(tblout_path)

    with open(output_path, 'w') as out_f:
        for record in SeqIO.parse(fasta_path, "fasta"):
            seqid = record.id
            seq = str(record.seq)
            families = hits.get(seqid, [])
            for family in families:
                entry = {
                    "seq": seq,
                    "family": family,
                    "seqid": seqid
                }
                out_f.write(json.dumps(entry) + "\n")

def write_family_fastas(json_file, output_dir="bprna_train_families", min_sequences=10):
    """
    Reads a JSONL file with RNA sequences and their families, and writes each family
    with at least `min_sequences` sequences to a separate FASTA file.

    Parameters:
    - jsonl_file (str): Path to the input JSONL file.
    - output_dir (str): Directory to save the FASTA files.
    - min_sequences (int): Minimum number of sequences required to create a FASTA file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to hold sequences grouped by family
    family_to_records = defaultdict(list)

    # Read and process the JSONL file
    with open(json_file, 'r') as infile:
        for line in infile:
            entry = json.loads(line)
            seq = entry.get("seq")
            seqid = entry.get("seqid")
            families = entry.get("family")

            # Ensure families is a list
            if isinstance(families, str):
                families = [families]

            for family in families:
                record = SeqRecord(Seq(seq), id=seqid, description="")
                family_to_records[family].append(record)

    # Write sequences to FASTA files for families with enough sequences
    for family, records in family_to_records.items():
        if len(records) >= min_sequences:
            filename = f"{family}.fasta"
            filepath = os.path.join(output_dir, filename)
            SeqIO.write(records, filepath, "fasta")
            print(f"Wrote {len(records)} sequences to {filepath}")


def run_cmsearch(train_file, output_folder, cpu_count=16):
    print(f"Please wait, running cmsearch...")
    command = [
    "/home/jc1417/anaconda3/envs/omnigenome3/bin/cmsearch", "--cpu",
    str(cpu_count), "--tblout",
    output_folder+"/rfam_hits.tblout", "--noali",
    rfam_covariance_model_path,
    str(train_file) # train.fasta
    ]
    subprocess.run(command, check=True)


def parse_cmscan_tblout(tbl_path):
    family_map = {}
    with open(tbl_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            target_name = parts[0]
            query_name = parts[2]
            family_map[query_name] = target_name
    return family_map


def remove_non_rfam_families(fasta_files_path):
    fasta_files = [x for x in os.dirlist(fasta_files_path) if x[-6:] == ".fasta"]
    for fasta_path in fasta_files:
        sequences = list(SeqIO.parse(fasta_path, "fasta"))
        
        family_map = parse_cmscan_tblout("cmscan_results.tbl")
        
        first_id = sequences[0].id
        ref_family = family_map.get(first_id)
        if not ref_family:
            print(f"No family found for first sequence ({first_id})")
            continue
        
        filtered_seqs = [seq for seq in sequences if family_map.get(seq.id) == ref_family]
        
        with open(fasta_path, "w") as out_f:
            FastaIO.FastaWriter(out_f, wrap=None).write_file(filtered_seqs)


def compute_distribution(seq, alphabet="ACGT"):
    count = Counter(seq)
    total = sum(count.get(base, 0) for base in alphabet)
    if total == 0:
        return np.full(len(alphabet), 1/len(alphabet))  # fallback to uniform
    return np.array([count.get(base, 0) / total for base in alphabet])


def min_pairwise_jsd(seqs):
    dists = [compute_distribution(str(seq.seq)) for seq in seqs]
    jsds = [jensenshannon(d1, d2) for i, d1 in enumerate(dists) for j, d2 in enumerate(dists) if i < j]
    return min(jsds) if jsds else 1.0


def remote_blastn_find_homologs_subseq_only(query_seqs, output_xml, output_dir, max_seqs=50, min_seqs=5, evalue=1e-3, jsd_threshold=.1):
    """
    Runs remote BLASTN against NCBI nt using Biopython.
    Saves the aligned subsequences only, no Entrez fetch.
    """

    if not os.path.isfile(output_xml):
    
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as query_file:
            for i, seq in enumerate(query_seqs):
                query_file.write(f">query_{i}\n{seq}\n")
            query_path = query_file.name
    
        print(f"Running local BLASTN on query...")
    
        # Step 2: Run local blastn using subprocess
        blastn_cmd = [
            ncbi_blast_path,
            "-task", "megablast",
            "-query", query_path,
            "-db", nt_database_path,
            "-out", output_xml,
            "-outfmt", "5",  # XML
            "-evalue", str(evalue),
            "-max_target_seqs", str(max_seqs),
            "-num_threads", "29",
            "-dust", "no",
            "-soft_masking", "false",
            "-max_hsps", "1",
        ]
    
        subprocess.run(blastn_cmd, check=True)
        print(f"Saved BLAST XML results to {output_xml}")
    with open(output_xml) as xml_handle:
        records = list(NCBIXML.parse(xml_handle))
    for j, blast_record in enumerate(records):
        subseq_records = []
        query_id = f"{j}"
        for alignment in blast_record.alignments:
            for i, hsp in enumerate(alignment.hsps):
                # Take the aligned subject subsequence 
                aligned_seq = hsp.sbjct.replace('-', '')  # gaps out
                subseq_len = len(aligned_seq)
    
                # Filter by length of aligned region
                if 20 <= subseq_len <= 4096:
                    subseq_record = SeqRecord(
                        Seq(aligned_seq),
                        id=str(query_id)+f"{i}_",
                        description=""   # f"len={subseq_len} identity={hsp.identities} evalue={hsp.expect}"
                    )
                    subseq_records.append(subseq_record)
                min_jsd = min_pairwise_jsd(subseq_records)
                if min_jsd < jsd_threshold:
                    # Seq too similar to add, must be 5% different to previous seqs
                    continue
                if len(subseq_records) > max_seqs:
                    break
            if len(subseq_records) > max_seqs:
                break
        

        if len(subseq_records) < min_seqs:
            print(f"Discarding result: only {len(subseq_records)} subsequences found (< 5).")
            continue
            
        print(f"Kept {len(subseq_records)} aligned subsequences between 20 and 4096 bp.")
        os.makedirs(output_dir, exist_ok=True)
        SeqIO.write(subseq_records, output_dir+f"{query_id}_homologs.fasta", "fasta")
    print(f"Wrote homologs in {output_dir}")


def mask_remaining_seqs(masked_data, old_seqs, output_json_path, restricted_nucleotides, mask_target_prop=0.15):
    masked_old_seqs = [x["original_sequence"].replace("U","T") for x in masked_data]
    new_masked_seqs = []
    for i, seq in enumerate(old_seqs):
        seq = seq.replace("U","T").upper()
        if seq in new_masked_seqs:
            # prevent dupes
            continue
        if seq in masked_old_seqs:
            index = masked_old_seqs.index(seq)
            one_masked_data = masked_data[index]
            one_masked_data["masked_sequence"] = one_masked_data["masked_sequence"].replace("<mask>","X")
            mask_prop = one_masked_data["masked_sequence"].count("X") / len(one_masked_data["original_sequence"])
            if len(one_masked_data["masked_sequence"]) != len(one_masked_data["original_sequence"]):
                    print("length mismatch before masking adjustment")
            if mask_prop > mask_target_prop:
                candidate_indices = [i for i, c in enumerate(one_masked_data["masked_sequence"]) if c == "X"]
                candidate_indices = [x for x in candidate_indices if x < len(one_masked_data["original_sequence"])]
                num_to_mask = int(math.floor(len(candidate_indices) * -(mask_target_prop - mask_prop)))
                indices_to_change = random.sample(candidate_indices, num_to_mask)
                one_masked_data["masked_sequence"] = list(one_masked_data["masked_sequence"])
                one_masked_data["original_sequence"] = list(one_masked_data["original_sequence"])
                for idx in indices_to_change:
                    one_masked_data["masked_sequence"][idx] = one_masked_data["original_sequence"][idx]
                one_masked_data["masked_sequence"] = "".join(one_masked_data["masked_sequence"])
                one_masked_data["original_sequence"] = "".join(one_masked_data["original_sequence"])
                if len(one_masked_data["masked_sequence"]) != len(one_masked_data["original_sequence"]):
                    print("length mismatch after masking adjustment (higher to lower)")
            else:
                candidate_indices = [i for i, c in enumerate(one_masked_data["masked_sequence"]) if c != "X"]
                needed = int(math.floor(len(candidate_indices) * (mask_target_prop - mask_prop)))
                indices_to_change = random.sample(candidate_indices, needed)
                one_masked_data["masked_sequence"] = list(one_masked_data["masked_sequence"])
                for idx in indices_to_change:
                    if idx+1 not in restricted_nucleotides:
                        one_masked_data["masked_sequence"][idx] = "X"
                one_masked_data["masked_sequence"] = "".join(one_masked_data["masked_sequence"])
            one_masked_data["masked_sequence"] = one_masked_data["masked_sequence"].replace("X","<mask>")

            
            new_masked_seqs.append(one_masked_data)
        else:
            continue
        
        
    with open(output_json_path, "w") as out_f:
        for entry in new_masked_seqs:
            out_f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(new_masked_seqs)} masked sequences to {output_json_path}")
    return masked_data


def adjust_seq_masking(masked_data, old_seq, restricted_nucleotides, mask_target_prop=0.15):
    masked_old_seq = old_seq.replace("U","T")
    masked_data = masked_data.replace("U","T")
    masked_data = masked_data.replace("<mask>","X")
    mask_prop = masked_data.count("X") / len(masked_data)
    if len(masked_data) != len(masked_old_seq):
        print("length mismatch before masking adjustment")
    if mask_prop > mask_target_prop:
        candidate_indices = [i for i, c in enumerate(masked_data) if c == "X"]
        candidate_indices = [x for x in candidate_indices if x < len(masked_data)]
        target_num_masks = int(math.floor(len(masked_data) * mask_target_prop))
        num_to_unmask = len(candidate_indices) - target_num_masks
        if num_to_unmask > 0:
            indices_to_change = random.sample(candidate_indices, num_to_unmask)
            masked_data = list(masked_data)
            masked_old_seq = list(masked_old_seq)
            for idx in indices_to_change:
                masked_data[idx] = masked_old_seq[idx]
    
            masked_data = "".join(masked_data)
            masked_old_seq = "".join(masked_old_seq)
            if len(masked_data) != len(masked_old_seq):
                print("length mismatch after masking adjustment (higher to lower)")
    else:
        candidate_indices = [i for i, c in enumerate(masked_data) if c != "X"]
        needed = int(math.floor(len(candidate_indices) * (mask_target_prop - mask_prop)))
        masked_data = list(masked_data)
        attempts = 0
        successes = 0
        max_attempts = len(candidate_indices) * 5  # avoid infinite loop
        
        while successes < needed and attempts < max_attempts:
            idx = random.choice(candidate_indices)
            attempts += 1
            if idx not in restricted_nucleotides and masked_data[idx] != "X":
                masked_data[idx] = "X"
                successes += 1

        masked_data = "".join(masked_data)

    return masked_data
