# 🧬 PhyloAug

**PhyloAug** is a genomic data augmentation tool designed to generate novel RNA structural datasets using phylogenetic and structural information. It enables the creation of augmented datasets that improve the performance of structure prediction models.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [Infernal](#infernal)
  - [Rfam](#rfam)
  - [PAML](#paml)
  - [NCBI BLAST](#ncbi-blast)
  - [nt Database](#nt-database)
- [Run Model](#run-model)
  - [Install Python Dependencies](#install-required-python-packages)
  - [Prepare Configuration](#prepare-configuration)
  - [Train the Model](#train-the-model)
- [Dataset Structure](#dataset-structure)
- [Citation](#citation)
- [Contributing](#contributing)

---

## Overview

PhyloAug leverages:
- **Phylogenetic profiles**
- **Secondary structural conservation**
- **Statistical substitution models**

to generate biologically meaningful augmented datasets suitable for machine learning tasks.

To generate new augmented datasets, see [Setup Instructions](#setup-instructions).  
To reproduce the results of the experiments, see [Run Model](#run-model).

---

## Requirements

To run PhyloAug, the following software and databases are required:

- [Infernal](http://eddylab.org/infernal/)
- [Rfam](https://rfam.org/)
- [PAML]([https://github.com/abacus-gene/paml](https://github.com/abacus-gene/paml))
- [NCBI BLAST+](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
- [NCBI `nt` database](ftp://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/nt.gz)

---

## Setup Instructions

### Infernal

```bash
wget http://eddylab.org/infernal/infernal.tar.gz
tar -xvzf infernal.tar.gz
cd infernal-*
./configure && make && sudo make install
```

### Rfam

```bash
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz
gunzip Rfam.cm.gz
cmpress Rfam.cm
```

### PAML

```bash
git clone https://github.com/abacus-gene/paml.git
cd paml/src
make
```

### NCBI BLAST

```bash
sudo apt-get install ncbi-blast+
```

### nt Database

The nt database is over 900GB, make sure you have adequate storage for this to download and unzip! (Recommended: 1.5TB)

[NCBI `nt` database](ftp://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/nt.gz)

Ensure the downloaded database is accessible by your BLAST installation.

---

## Run Model

### 1. Install Required Python Packages

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install omnigenbench transformers datasets huggingface_hub
```

> CUDA-compatible GPU and drivers recommended for faster training.

---

### 2. Prepare Configuration

Edit the `config.py` file to use the desired augmented dataset:

Available datasets:
- **Tasks**: `Archive2`, `bpRNA`, `rnastralign`
- **Augmentation levels**: `1`, `2`, `4`, `8`, `12`

```python
# Example: config.py
TRAINING_DATASET = "augmented/bpRNA_aug_4.json"
```

---

### 3. Train the Model

```bash
autotrain -m "model pathway or huggingface URL like: yangheng/OmniGenome-52M" -d "pathway to config.py file e.g., RNA-SSP-Archive2/config.py"
```

---

## Dataset Structure

Each augmented dataset includes:
- Original RNA Data
- Augmented Input RNA sequences
- Corresponding Secondary structure annotations

---

## Citation

If you use PhyloAug in your research, please cite:

```Removed as to not break anonymity.```

---

## Contributing

We welcome contributions!  
Feel free to open an issue or submit a pull request to suggest improvements or extensions.
