# Data Folder

## Raw Dataset

This project uses the **GoEmotions** dataset by Google Research. Due to file size limitations, raw files are not uploaded to GitHub. Please run `scripts/01_load_data.py` to automatically download and generate the dataset.

---

### Dataset — GoEmotions (2020)
- **Paper:** Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.
- **arXiv:** https://arxiv.org/abs/2005.00547
- **HuggingFace:** https://huggingface.co/datasets/google-research-datasets/go_emotions
- **File name:** `go_emotions_raw.csv`
- **Place at:** `data/raw/go_emotions_raw.csv`

---

## Processed Dataset

After running `scripts/02_preprocess_data.py`, the cleaned dataset will be saved at:
`data/processed/combined_dataset.csv`

This file contains **58,008 Reddit comments** with the following columns:
- `text` — cleaned comment text
- `parrot_label` — one of: Joy, Love, Surprise, Anger, Sadness, Fear, Neutral
- `comment_length` — number of characters
- `word_count` — number of words
