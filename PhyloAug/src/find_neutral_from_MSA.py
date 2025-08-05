from math import log2
from collections import Counter
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import shutil
from tqdm import tqdm
import pexpect
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
import csv
import re
import json
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from src.GLOBAL_PATHS import *



def obtain_fasttree(msa_dir="bprna_train_families"):
    # Set the working directory (use '.' for current directory)
    msa_dir = Path(msa_dir)

    # Loop through all .aln files
    for msa_file in msa_dir.glob("*.aln"):
        print(f"Running FASTTREE on: {msa_file.name}")

        cmd = f"fasttree -nt {msa_file} > {str(msa_file)[:-4]}.tree"
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running FASTTREE on {msa_file.name}: {e}")


def mask_sequences_to_json_phylogenetic(df, phy_file, output_json_path="masked_train.json", mask_token="<mask>", neutral_prob_cutoff=0.9):
    """
    Mask bases in sequences based on prob_neutral threshold.
    - df: pandas DataFrame with columns [site, sequence_index, codon_pos, base, prob_neutral]
    - phy_file: input .phy file
    - output_json_path: where to write JSON lines
    """
    # If needed: convert .phy to .fasta
    # (Assumes your .phy is in FASTA format already, or converted externally)
    # If not, convert using BioPython AlignIO if required.

    alignment = list(SeqIO.parse(phy_file, "phylip"))

    results = []

    # Filter the DataFrame once for prob_neutral > 0.8
    mask_df = df[df["neutral_or_not"] == True]

    for record in alignment:
        # The sequence we want is always the first in the .aln file
        seq_id = record.id
        seq = list(str(record.seq))
        original_seq = "".join(seq).upper()
        masked_seq = seq[:]

        for _, row in mask_df.iterrows():
            site = int(row["sequence_index"]) - 1
            masked_seq[site] = mask_token

        result_entry = {
            "file_name": phy_file.stem,
            "seq_id": seq_id,
            "original_sequence": original_seq.replace("-",""),
            "masked_sequence": "".join(masked_seq).replace("-","")
        }
        return result_entry


def deduplicate_msa(input_aln, output_aln):
    """
    Fix duplicate IDs in an alignment:
      - Remove records with identical sequence & same ID.
      - If duplicate ID but different sequence, rename.
    Writes cleaned alignment to output_aln.
    """
    records = list(SeqIO.parse(input_aln, "fasta"))

    unique_records = []
    seen_seqs = {}
    id_counts = {}

    for record in records:
        seq_str = str(record.seq).upper()
        if record.id in seen_seqs:
            if seen_seqs[record.id] == seq_str:
                continue
            else:
                id_counts[record.id] = id_counts.get(record.id, 1) + 1
                new_id = f"{record.id}{id_counts[record.id]}"
                record.id = new_id
                record.name = ""
                record.description = ""
                seen_seqs[new_id] = seq_str
        else:
            seen_seqs[record.id] = seq_str

        unique_records.append(record)

    SeqIO.write(unique_records, output_aln, "fasta")


def compute_site_frequency(alignment):
    seq_count = len(alignment)
    freqs = []
    for col in zip(*alignment):
        count = Counter(col)
        valid = sum(1 for base in col if base not in ["-", "N", "X"])
        freqs.append(valid)
    return freqs

