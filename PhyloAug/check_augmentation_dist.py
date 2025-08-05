import json
import copy
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import logomaker
import matplotlib
matplotlib.use('Agg')
from Bio import SeqIO


class RNALogoAnalyzer:
    def __init__(self, sequences, restrict_to=False):
        self.sequences = sequences
        self.count_df = self.to_count_df()
        self.prob_df = self.to_prob_df()

    def read_sequences(self, restrict_to=False):
        lines = []
        with open(self.json_path, "r") as file:
            for line in file.readlines():
                lines.append(json.loads(line.strip()))

        if restrict_to:
            og_lines = []
            with open(restrict_to, "r") as file:
                for line in file.readlines():
                    og_lines.append(json.loads(line.strip()))
            lines2 = lines
            lines = []
            for line in lines2:
                if line in og_lines:
                    continue
                lines.append(line)
            og_structs = [x["label"] for x in og_lines]
            return [x["seq"].replace("T","U") for x in lines if x["label"] in og_structs]
        else:
            return [x["seq"].replace("T","U") for x in lines]

    def to_count_df(self):
        max_len = max(len(s) for s in self.sequences)
        padded = [s.ljust(max_len, 'N') for s in self.sequences]
        bases = ['A', 'C', 'G', 'U', 'N']
        counts = {b: [] for b in bases}
        for pos in range(max_len):
            col = [s[pos] for s in padded]
            for base in 'ACGU':
                counts[base].append(col.count(base))
            counts['N'].append(len(col) - sum(counts[b][-1] for b in 'ACGU'))
        return pd.DataFrame(counts)

    def to_prob_df(self):
        return self.count_df.div(self.count_df.sum(axis=1), axis=0)

    def write_fasta(self, out_path):
        with open(out_path, 'w') as f:
            for i, seq in enumerate(self.sequences):
                f.write(f">seq{i}\n{seq}\n")

    def plot_logo(self, title=None, backend="logomaker"):
        if title is None:
            title = f"Logo: {self.json_path}"
        prob_df = self.prob_df[['A', 'C', 'G', 'U']]

        if backend == "logomaker":
            logo = logomaker.Logo(prob_df)
            # style setup...
        elif backend == "weblogo":
            self.write_fasta("_tmp.fasta")
            os.system(f"weblogo -f _tmp.fasta -o {title}.png --format png")
        else:
            raise ValueError(f"Unknown backend: {backend}")
        if hasattr(self, 'jsd') and hasattr(self, 'cosine'):
            plt.title(f'{title} (JSD: {self.jsd:.4f}, Cosine: {self.cosine:.4f})')
        else:
            plt.title(title)
        plt.savefig(f"{title}.pdf", dpi=300)
        plt.show()

    def compare_distributions(self, prob_df1, prob_df2):
        def js_div(p, q):
            p, q = np.asarray(p), np.asarray(q)
            m = 0.5 * (p + q)
            def kl(a, b):
                mask = a > 0
                return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
            return 0.5 * (kl(p, m) + kl(q, m))

        def cosine(p, q):
            p, q = np.asarray(p), np.asarray(q)
            return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))

        L = min(len(prob_df1), len(prob_df2))
        jsd_list = [js_div(prob_df1.iloc[i], prob_df2.iloc[i]) for i in range(L)]
        cos_list = [cosine(prob_df1.iloc[i], prob_df2.iloc[i]) for i in range(L)]
        self.jsd = np.mean(jsd_list)
        self.cosine = np.mean(cos_list)
        return np.mean(jsd_list), np.mean(cos_list)


def read_json_file(filename):
    lines = []
    with open(filename, "r") as file:
        for line in file.readlines():
            lines.append(json.loads(line.strip()))
    return lines


def get_sequences_with_struct(lines, struct):
    seqs = []
    for line in lines:
        if line["label"] == struct:
            seqs.append(line["seq"].replace("T","U"))
    return seqs


def get_non_aug_seqs_with_struct(lines, struct, filepath="RNA-SSP-rnastralign_train_families/"):
    matching_seqs = get_sequences_with_struct(lines, struct)
    original_seqs = [x["seq"].replace("T","U") for x in lines]
    for seq in matching_seqs:
        original_homologs = []
        idx = original_seqs.index(seq)
        try:
            with open(filepath+str(idx)+"_homologs.fasta", "r") as file:
                for record in SeqIO.parse(file, "fasta"):
                    original_homologs.append(str(record.seq).replace("T","U"))
            if any(seq in original_seqs for seq in original_homologs):
                return original_homologs
        except: # No homologs found
            return None
    return None

if __name__ == "__main__":

    non_aug_data = read_json_file("RNA-SSP-rnastralign/train.json")
    aug_1_data = read_json_file("RNA-SSP-rnastralign/randomly_augmented_train_aug1.json")
    aug_2_data = read_json_file("RNA-SSP-rnastralign/randomly_augmented_train_aug2.json")
    aug_4_data = read_json_file("RNA-SSP-rnastralign/randomly_augmented_train_aug4.json")
    aug_8_data = read_json_file("RNA-SSP-rnastralign/randomly_augmented_train_aug8.json")
    aug_12_data = read_json_file("RNA-SSP-rnastralign/randomly_augmented_train_aug12.json")

    # Group data by structure
    non_aug_data_structs = [x["label"] for x in non_aug_data]
    # pass in sequences to RNALogoAnalyzer to analyse the distribution
    for aug_num in [1, 2, 4, 8, 12]:
        jsd_vals = []
        cos_vals = []
        aug_data = read_json_file(f"RNA-SSP-rnastralign/randomly_augmented_train_aug{aug_num}.json")
        skipped_homologs = 0
        for struct in non_aug_data_structs:
            non_aug_seqs = get_non_aug_seqs_with_struct(non_aug_data, struct)
            if non_aug_seqs == None:
                skipped_homologs += 1
                continue
            aug_seqs = get_sequences_with_struct(aug_data, struct)
            if len(aug_seqs) < 1:
                continue
            if aug_num == 0:
                path = 'RNA-SSP-rnastralign/train.json'
            else:
                path = f"RNA-SSP-rnastralign/randomly_augmented_train_aug{aug_num}.json"
            orig = RNALogoAnalyzer(non_aug_seqs)
            aug = RNALogoAnalyzer(aug_seqs)
            jsd, cos = aug.compare_distributions(orig.prob_df, aug.prob_df)
            jsd_vals.append(jsd)
            cos_vals.append(cos)
        print(f"{aug_num}: JSD={np.mean(jsd_vals):.4f}, Cosine={np.mean(cos_vals):.4f}")
        print(f"Skipped sequences due to lack of homologs: {skipped_homologs}")