import os
import subprocess
import tqdm as tqdm
from Bio import SeqIO
from src.GLOBAL_PATHS import *

def has_5_seqs(fasta_path):
    count = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    return count >= 5


def run_mafft_on_rna_families(input_folder, threads=24):
    # input_folder = Folder containing .fasta files (update this path as needed)

    # Get all .fasta files in the specified folder
    fasta_files = [f for f in os.listdir(input_folder) if f.endswith('.fasta')]

    # Progress bar for all files
    for fasta_file in tqdm.tqdm(fasta_files, desc="Aligning FASTA files", total=len(fasta_files)):
        input_path = os.path.join(input_folder, fasta_file)
        output_path = os.path.join(input_folder, fasta_file.replace('.fasta', '_MSA.aln'))
        if has_5_seqs(input_path):
    
            cmd = f'{mafft_path} --auto --inputorder --nuc --nomemsave --large --thread {threads} {input_path} > {output_path}'
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True)
        else:
            print(f"Skipped {input_path} due to insignificant sequence count")


def run_mafft_on_rna_fasta(fasta_file, input_folder, threads=24):
    input_path = fasta_file
    output_path = fasta_file.with_name(fasta_file.stem + '_MSA.aln')
    if has_5_seqs(input_path):

        cmd = f'{mafft_path} --auto --inputorder --nuc --nomemsave --large --thread {threads} {input_path} > {output_path}'
        # print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"Skipped {input_path} due to insignificant sequence count")