# IE 423 Term Project — Hate Speech in Social Media: Detecting Polarized Tweets Using Machine Learning

## Team Members
- Basil Mohammad A. Sadlah — 123203115
- Saleh Rami (Moh'd Saleh) Yaish — 121203025
- Parsa Badiee — 120203094
- Mohammed Saleh Mohammed Al-Hamami — 120203098

## Dataset
We use three publicly available, peer-reviewed hate speech datasets:

| Dataset | Source | Year |
|---|---|---|
| Hate Speech & Offensive Language | Davidson et al., ICWSM 2017 | 2017 |
| Call Me Sexist But (CMSB) | Samory et al., ICWSM 2021 | 2021 |
| MMHS150K | Gomez et al., IEEE/CVF WACV 2020 | 2020 |

After preprocessing and merging, the final dataset contains **61,945 tweets** across **6 categories**.

## Project Objective
To build a machine learning classifier that detects and categorizes hate speech in tweets into the following classes: Racist, Sexist, Offensive Language, Other Hate, Religion, and Not Hate.

## Repository Structure
```
|
├── README.md
├── requirements.txt
|
├── data/
│   ├── raw/
|   |   └── dataset file(s) or dataset link instructions
│   ├── processed/
|   |   └── cleaned data outputs
│   └── README.md
|
├── scripts/
│   ├── 01_load_data.py
│   ├── 02_preprocess_data.py
│   └── 03_basic_eda.py
|
├── outputs/
│   ├── figures/     
│   └── tables/
|
└── docs/
    └── ResearchProposalPreprocessing.md
```
## Installation
```bash
pip install -r requirements.txt
```

## Running the Scripts
Run the scripts in order from the root of the repository:
```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```

> **Note:** Place the raw dataset files inside `data/raw/` before running. See `data/README.md` for instructions.

## Main Proposal Document
See: `docs/ResearchProposalPreprocessing.md`
