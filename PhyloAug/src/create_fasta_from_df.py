# Re-import necessary modules after code reset
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from ast import literal_eval
from pathlib import Path
import json
import RNA
from Bio import AlignIO
from src.parse_rna_fam_data import adjust_seq_masking
from src.GLOBAL_PATHS import *



def mask_sequences_to_json(df, output_json_path="masked_sequences.json", mask_token="<mask>"):
    # df = pd.read_csv(mask_table_file)

    results = []

    for _, row in df.iterrows():
        aln_file = str(row[0])
        mask_infos = [
            literal_eval(cell) if isinstance(cell, str) else cell
            for cell in row[1:] if pd.notna(cell)
        ]

        aln_file = aln_file[:-8] + ".fasta"
        alignment_record = next(SeqIO.parse(aln_file, "fasta"))
        alignment = [alignment_record]
        for record in alignment:
            seq = list(str(record.seq))
            original_seq = ("".join(seq)).upper()
            masked_seq = adjust_seq_masking("".join(seq), original_seq, [])
            masked_seq = [x for x in masked_seq]
            allowed_info = []

            for mask_info in mask_infos:
                site = mask_info['site'] - 1  # 0-based
                if site >= len(seq):
                    continue

                # Prepare allowed base info
                allowed_bases = {base.upper(): prob for base, prob in mask_info.items() if
                                 base.upper() in ['A', 'C', 'G', 'U', 'N'] and prob > 0}
                allowed_info.append({str(site + 1): allowed_bases})  # convert back to 1-based for output

                # Mask the base
                masked_seq[site] = mask_token

            result_entry = {
                "file_name": aln_file,
                "seq_id": record.id,
                "original_sequence": original_seq,
                "masked_sequence": "".join(masked_seq),
                # "allowed_nucleotides_per_masked_position": allowed_info
            }

            results.append(result_entry)

    # Write to JSON format
    with open(output_json_path, "w") as out_f:
        for entry in results:
            out_f.write(json.dumps(entry) + "\n")

    return results


def add_secondary_structure_to_json(jsonl_file, structure_dict, output_jsonl_file):
    pairs = ["(",")","[","]","{","}","<",">","A","a","B","b","C","c","D","d","E","e","F","f","G","g"]
    with open(jsonl_file, "r") as infile, open(output_jsonl_file, "w") as outfile:
        for line in infile.readlines():
            record = json.loads(line.strip())
            seq = record.get("original_sequence").replace("U","T").upper()
            aug_seq = record.get("masked_sequence")
            aug_seq = aug_seq.replace("<mask>", "X")

            # Match the sequence exactly
            structure = structure_dict.get(seq.upper())
            if structure:
                record["structure"] = structure
                record["label"] = structure
            else:
                print("Failed to obtain true structure")
                print(seq)
                print(list(structure_dict.keys())[0])
                struct = RNA.fold(seq)[0]
                record["structure"] = struct
                record["label"] = struct
                structure = struct
            
            aug_seq = aug_seq.replace("X","<mask>")

            outfile.write(json.dumps(record) + "\n")

    print(f"Secondary structures added and saved to {output_jsonl_file}")


def add_label_to_json(jsonl_file, og_data_file, output_jsonl_file, structure_dict, mRNA=False):
    og_data = []
    with open(og_data_file, "r") as file:
        for line in file.readlines():
            og_data.append(json.loads(line.strip()))

    with open(jsonl_file, "r") as infile, open(output_jsonl_file, "w") as outfile:
        for line in infile.readlines():
            record = json.loads(line.strip())
            # print(record["original_sequence"].replace("-", "").replace("U","T").upper())
            seq = record.get("original_sequence")
            seq = seq.replace("-", "").replace("U","T").upper()
            masked_seq = record.get("masked_sequence")
            masked_seq = masked_seq.replace("-","")
            
            # Match the sequence exactly
            matching_seq = None

            matching_seq = int(record.get("seq_id")[:-2])
            if matching_seq == None:
                continue # do not use record if we cannot obtain the original label

            if matching_seq is not None and not mRNA:
                record["label"] = og_data[matching_seq]["label"]
            elif matching_seq is not None and mRNA:
                record["reactivity"] = og_data[matching_seq]["reactivity"]
                record["deg_Mg_pH10"] = og_data[matching_seq]["deg_Mg_pH10"]
                record["deg_Mg_50C"] = og_data[matching_seq]["deg_Mg_50C"]
            else:
                continue # do not use record if we cannot obtain the original label

            # Match the structure exactly
            structure = structure_dict.get(seq.upper())
            if type(structure) == str:
                structure_check = [x for x in structure if [".", "(", ")"] in x]
            else:
                structure_check = ""
            if len(structure_check) > 1:
                record["structure"] = structure
            else:
                record["structure"] = None  # if not found set to None so ViennaRNA can predict it

            outfile.write(json.dumps(record) + "\n")

    print(f"Secondary structures added and saved to {output_jsonl_file}")


def get_structure_lookup_dict(json_file_name):
    structure_lookup = {}
    with open(json_file_name, "r") as file:
        for line in file.readlines():
            line = json.loads(line.strip())
            if "label" in line.keys() and "seq" in line.keys():
                structure_lookup[line["seq"].upper().replace("U","T")] = line["label"]
            elif "seq" in line.keys():
                structure_lookup[line["seq"].upper().replace("U","T")] = line["structure"]
            elif "label" in line.keys():
                structure_lookup[line["sequence"].upper().replace("U","T")] = line["label"]
            else:
                structure_lookup[line["sequence"].upper().replace("U","T")] = line["structure"]
    return structure_lookup
