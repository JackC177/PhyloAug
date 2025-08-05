import subprocess
import os
from pathlib import Path
import pexpect
from ete3 import Tree
import re


def clean_newick(infile, outfile):
    # Fix mis-alignment in tree with ete3 and paml
    with open(infile) as fin:
        tree = fin.read().strip()

    tree = re.sub(r'\)\s*[0-9eE\.\-+]+\s*:', r'):', tree)
    # Remove internal node labels before ";" (end of tree)
    tree = re.sub(r'\)\s*[0-9eE\.\-+]+\s*;', r');', tree)

    with open(outfile, 'w') as fout:
        fout.write(tree + '\n')


def get_phylogenetic_trees_fasttree(msa_dir):
    # Loop through all .aln files
    msa_dir = Path(msa_dir)
    for msa_file in msa_dir.glob("*.aln"):
        print(f"Running FASTTREE on: {msa_file.name}")
        
        cmd = f"./FastTree -nt -gtr {msa_file} > {str(msa_file)[:-4]}.tree"
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running FASTTREE on {msa_file.name}: {e}")

        t = Tree(f"{str(msa_file)[:-4]}.tree")
        t.resolve_polytomy(recursive=True)
        t.write(outfile=f"{str(msa_file)[:-4]}.tree")
