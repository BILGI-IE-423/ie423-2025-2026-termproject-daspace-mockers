# IE 423 Term Project — Emotion Detection in Social Media: Classifying Reddit Comments Using Machine Learning

## Team Members
- Basil Mohammad A. Sadlah — 123203115
- Saleh Rami (Moh'd Saleh) Yaish — 121203025
- Parsa Badiee — 120203094
- Mohammed Saleh Mohammed Al-Hamami — 120203098

## Dataset
We use the **GoEmotions** dataset by Google Research:

| Dataset | Source | Year |
|---|---|---|
| GoEmotions | Demszky et al., ACL 2020 | 2020 |

After preprocessing, the final dataset contains **58,008 Reddit comments** across **7 emotion categories**.

## Project Objective
To build a machine learning classifier that detects and categorizes emotions in Reddit comments into the following classes based on Parrot's emotion framework: Joy, Love, Surprise, Anger, Sadness, Fear, and Neutral.

## Repository Structure
```
|
├── README.md
├── requirements.txt
|
├── data/
│   ├── raw/
|   |   └── go_emotions_raw.csv
│   ├── processed/
|   |   └── combined_dataset.csv
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

> **Note:** Raw data is automatically downloaded when running `01_load_data.py`. See `data/README.md` for details.

## Main Proposal Document
See: `docs/ResearchProposalPreprocessing.md`
