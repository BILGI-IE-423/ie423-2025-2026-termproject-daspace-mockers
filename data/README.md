# Data Folder

## Raw Datasets

This project uses three datasets. Due to file size limitations, raw files are not uploaded to GitHub. Please download them manually and place them inside `data/raw/`.

---

### Dataset 1 — Davidson Hate Speech & Offensive Language (2017)
- **Paper:** Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM 2017.
- **DOI:** https://doi.org/10.1609/icwsm.v11i1.14955
- **Download:** https://github.com/t-davidson/hate-speech-and-offensive-language
- **File name:** `labeled_data.csv`
- **Place at:** `data/raw/labeled_data.csv`

---

### Dataset 2 — Call Me Sexist But / CMSB (2021)
- **Paper:** Samory, M., Sen, I., Kohne, J., Flöck, F., & Wagner, C. (2021). "Call me sexist, but…": Revisiting Sexism Detection Using Psychological Scales and Adversarial Samples. ICWSM 2021.
- **DOI:** https://doi.org/10.1609/icwsm.v15i1.18085
- **Download:** https://doi.org/10.7802/2251 (GESIS Data Archive)
- **File name:** `sexism_data.csv`
- **Place at:** `data/raw/sexism_data.csv`

---

### Dataset 3 — MMHS150K (2020)
- **Paper:** Gomez, R., Gibert, J., Gomez, L., & Karatzas, D. (2020). Exploring Hate Speech Detection in Multimodal Publications. IEEE/CVF WACV 2020.
- **arXiv:** https://arxiv.org/abs/1910.03814
- **Download:** https://drive.google.com/file/d/1S9mMhZFkntNnYdO-1dZXwF_8XIiFcmlF/view
- **File name:** `MMHS150K_GT.json`
- **Place at:** `data/raw/MMHS150K_GT.json`

---

## Processed Dataset

After running `scripts/02_preprocess_data.py`, the cleaned and merged dataset will be saved at:

`data/processed/combined_dataset.csv`

This file contains **61,945 tweets** with the following columns:
- `tweet_text` — cleaned tweet text
- `label` — one of: Racist, Sexist, Offensive Language, Other Hate, Religion, Not Hate
- `tweet_length` — number of characters
- `word_count` — number of words
