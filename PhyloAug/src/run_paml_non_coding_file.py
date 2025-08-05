import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import re
import pandas as pd
import math
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from src.run_mafft_rna_families import run_mafft_on_rna_fasta
from src.find_neutral_from_MSA import deduplicate_msa
import json
from Bio import AlignIO
from collections import Counter
import random
from src.parse_rna_fam_data import adjust_seq_masking
import os
from src.GLOBAL_PATHS import *


def mask_sequences_to_json_phylogenetic(df, phy_file, output_json_path="masked_train.json", mask_token="<mask>"):
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

    mask_df = df[df["neutral_or_not"] == True]

    for record in alignment:
        # The sequence we want is always the first in the .aln file
        seq_id = record.id
        seq = list(str(record.seq))
        original_seq = "".join(seq).upper()
        masked_seq = seq[:]

        for _, row in mask_df.iterrows():
            site = int(row["sequence_index"]) - 1
            if masked_seq[site] == "-":
                continue
            masked_seq[site] = mask_token

        result_entry = {
            "file_name": phy_file.stem,
            "seq_id": seq_id,
            "original_sequence": original_seq.replace("-","").replace("U","T"),
            "masked_sequence": "".join(masked_seq).replace("-","").replace("U","T")
        }
        return result_entry


def write_ctl_file(run_dir, model_type, loose_paml=False):
    """Generate baseml.ctl for neutral or conserved model."""
    ctl_path = run_dir / "baseml.ctl"
    with open(ctl_path, "w") as f:
        f.write(f"""seqfile = seqfile
treefile = treefile""")
        f.write("\noutfile = rst")
        f.write(f"""\nrunmode = 0
model = 0           * REV
clock = 0
RateAncestor = 1
fix_blength = 0
getSE = 0
cleandata = 0
Mgene = 0
""")
        if loose_paml:
            f.write("\nfix_alpha = 0\nalpha = 0.2\nncatG = 12\n")
        else:
            f.write("\nfix_alpha = 0\nalpha = 1\nncatG = 10\n")


def extract_lnL(baseml_out_path):
    """Extract log-likelihood from baseml.out"""
    with open(baseml_out_path) as f:
        for line in f:
            if "ln Lmax" in line:
                return float(line.split("=")[-1].strip())
    raise ValueError(f"lnL not found in {baseml_out_path}")


def parse_rst_posteriors(rst_path, conserved_threshold=0.7, neutral_upper_threshold=1.3):
    """
    Parse PAML baseml .rst file to extract site-wise posterior rate estimates
    and classify them into 'conserved', 'neutral', or 'ambiguous'.

    Args:
        rst_path (Path): Path to the .rst file from baseml.
        conserved_threshold (float): Upper limit to classify site as conserved.
        neutral_threshold (float): Lower limit to classify site as neutral.

    Returns:
        pd.DataFrame: One row per site with classification.
    """
    with open(rst_path) as f:
        lines = f.readlines()

    # Find start of site-specific rates section
    try:
        start_idx = next(i for i, line in enumerate(lines) if "Rate (posterior mean" in line)
    except StopIteration:
        print(f"Couldn't find site rate section in {rst_path}")
        return None

    # Data lines follow the header
    site_lines = lines[start_idx + 2:]

    site_info = []
    for line in site_lines:
        if not re.match(r"^\s*\d+", line):
            break  # End of section

        parts = line.strip().split()
        if len(parts) < 5:
            continue  # Skip malformed lines

        site = int(parts[0])
        freq = int(parts[1])
        data = parts[2]
        rate = float(parts[3])
        category = int(parts[4])

        if conserved_threshold <= rate <= neutral_upper_threshold:
            label = "neutral"
        elif rate < conserved_threshold:
            label = "conserved"
        elif rate > neutral_upper_threshold:
            label = "fast"
        else:
            label = "ambiguous"

        site_info.append({
            "site": site,
            "freq": freq,
            "sequence": data,
            "rate": rate,
            "rate_category": category,
            "label": label
        })

    return pd.DataFrame(site_info)


def get_rfam_family(fasta_file, rfam_cm_path=rfam_covariance_model_path, cmscan_bin=f"{infernal_path}/src/cmscan"):
    tblout = fasta_file.with_suffix(".cmscan.tbl")
    subprocess.run([
        cmscan_bin,
        "--noali", "--tblout", str(tblout),
        rfam_cm_path, str(fasta_file)
    ], stdout=subprocess.DEVNULL)
    
    with open(tblout) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            return parts[0]  # family accession like RF00001
    return None


def fetch_cm_model(family_id, rfam_cm_path=rfam_covariance_model_path, out_path="rfam_model.cm", cmfetch_bin=f"{infernal_path}/src/cmfetch"):
    with open(out_path, "w") as f:
        subprocess.run([cmfetch_bin, rfam_cm_path, family_id], stdout=f)


def run_cmalign(cm_path, fasta_file, out_sto):
    subprocess.run([f"{infernal_path}/src/cmalign", "--outformat", "Stockholm", cm_path, fasta_file], stdout=open(out_sto, "w"))


def extract_conserved_from_stockholm(stockholm_path, threshold=0.70):
    alignment = AlignIO.read(stockholm_path, "stockholm")
       
    aln_len = alignment.get_alignment_length()
    
    conserved_positions = []
    for i in range(aln_len):
        column = [record.seq[i] for record in alignment]
        freqs = Counter(column)
        if '-' in freqs:
            del freqs['-']
        if not freqs:
            continue
        base, freq = freqs.most_common(1)[0]
        conservation_ratio = freq / len(alignment)
        if conservation_ratio >= threshold:
            conserved_positions.append(i)
                
    return conserved_positions


def run_paml(base_dir, output_json_path, no_rfam=False):
    """Run both neutral and conserved baseml models per alignment and compute LRT."""
    fasta_files = list(base_dir.glob("*.fasta"))
    fasta_files = [x for x in fasta_files if "gapped" not in x.name]
    results = []

    for fasta_file in tqdm(fasta_files, desc="Running PAML on alignments", total=len(fasta_files)):
        aln_file = fasta_file.with_name(fasta_file.stem + "_MSA.aln")
        possible_msa = next(base_dir.glob(str(aln_file)), None)
        if not possible_msa or not possible_msa.exists():
            run_mafft_on_rna_fasta(fasta_file, base_dir)
            deduplicate_msa(aln_file, aln_file)
            aln = AlignIO.read(str(aln_file), "fasta")

        if len(aln) < 5:
            continue  # Skip underpowered MSAs

        for rec in aln:
            rec.seq = rec.seq.upper().back_transcribe().replace("U", "T")
        AlignIO.write(aln, aln_file, "fasta")

        phy_file = aln_file.with_suffix(".phy")
        AlignIO.convert(aln_file, "fasta", phy_file, "phylip-sequential")
        base_name = phy_file.stem
        possible_tree = next(base_dir.glob(f"{base_name}.tree"), None)

        if not possible_tree or not possible_tree.exists():
            print(f"No matching tree file for {aln_file.name}, skipping.")
            continue

        model_type = "conserved"
        run_dir = base_dir / f"paml_run_{base_name}_{model_type}"
        print("checking for: ", run_dir)
        if not os.path.exists(run_dir):
            print("Running PAML")
            run_dir.mkdir(exist_ok=True)
    
            shutil.copy(aln_file, run_dir / "seqfile")
            shutil.copy(possible_tree, run_dir / "treefile")
            write_ctl_file(run_dir, model_type)
    
            subprocess.run(paml_baseml_path, shell=True, cwd=run_dir, 
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        rst_file = run_dir / "rates"
        out_file = run_dir / "rst"
        if not out_file.exists():
            print(f"Missing baseml output for {model_type} model")
            continue

        # Parse site-wise estimates
        rst_df = parse_rst_posteriors(rst_file)
        if rst_df is None:
            print("PAML run failed or incomplete.")
            continue # error due to user running and cancelling mid-way
        df = pd.DataFrame({
            "sequence_index": rst_df["site"],
            "neutral_or_not": rst_df["label"] == "neutral"
        })
        try:
            result = mask_sequences_to_json_phylogenetic(df, phy_file)
        except:
            print("PAML run incomplete")
            continue

        if result == None:
            continue
            
        cleaned_path = fasta_file.with_suffix(".ungapped.fasta")
        with open(cleaned_path, "w") as out_f:
            for record in SeqIO.parse(fasta_file, "fasta"):
                record.seq = record.seq.replace("-", "")
                SeqIO.write(record, out_f, "fasta")

        if not no_rfam:
            print("Rfam for conserved nucleotides")
            rfam_id = get_rfam_family(cleaned_path)
            if rfam_id:
                cm_path = base_dir / f"{rfam_id}.cm"
                fetch_cm_model(rfam_id, out_path=cm_path)
                sto_path = base_dir / f"{rfam_id}_align.sto"
                run_cmalign(cm_path, cleaned_path, sto_path)
                conserved_sites = extract_conserved_from_stockholm(sto_path)
                for i, row in rst_df.iterrows():
                    if row["label"] in ["conserved", "fast"]:
                        conserved_sites.append(row["site"])
                result["masked_sequence"] = result["masked_sequence"].replace("<mask>","X")
                x_indexes = [i for i, char in enumerate(result["masked_sequence"]) if char == "X"]
    
    
                # Remove any wrongly masked sites
                for i in conserved_sites:
                    if i < len(result["masked_sequence"]):
                        if result["masked_sequence"][i] == "X":
                            result["masked_sequence"] = (
                                result["masked_sequence"][:i] + result["original_sequence"][i] + result["masked_sequence"][i + 1:]
                            )
                x_indexes = [i for i, char in enumerate(result["masked_sequence"]) if char == "X"]
                print(f"Neutral: {sum(1 for x in result['masked_sequence'] if x == "X")/len(result['masked_sequence'])}")
                # Random masking without forgetting the conserved_sites
                result["masked_sequence"] = result["masked_sequence"].replace("X","<mask>")
                result["masked_sequence"] = adjust_seq_masking(result["masked_sequence"], result["original_sequence"], conserved_sites)
                print(f"Neutral: {sum(1 for x in result['masked_sequence'] if x == "X")/len(result['masked_sequence'])}")
                result["masked_sequence"] = result["masked_sequence"].replace("X","<mask>")
        else:
            conserved_sites = rst_df[rst_df["label"].isin(["conserved", "fast"])]["site"].tolist()
            if len(conserved_sites) == len(rst_df["site"].tolist()):
                conserved_sites = []  # create random samples
            result["masked_sequence"] = result["masked_sequence"].replace("X","<mask>")
            result["masked_sequence"] = adjust_seq_masking(result["masked_sequence"], result["original_sequence"], conserved_sites)
            print(f"Neutral: {sum(1 for x in result['masked_sequence'] if x == "X")/len(result['masked_sequence'])}")
            result["masked_sequence"] = result["masked_sequence"].replace("X","<mask>")

        results.append(result)

    with open(output_json_path, "w") as out_f:
        for entry in results:
            out_f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(results)} masked sequences to {output_json_path}")

    return results
