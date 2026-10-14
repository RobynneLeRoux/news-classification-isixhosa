# News Classification in English and isiXhosa

Multinomial logistic regression, built in PyTorch, that sorts news articles into topic categories for **English and isiXhosa** using the [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news) dataset.

The interesting question: does the best text representation for English also work for a lower-resource African language? It doesn't.

## Results

| Language | Best features | Validation accuracy | Notes |
|---|---|---|---|
| English | Bag-of-words | 87.5% (test 88.7%) | All three feature types performed well |
| isiXhosa | Binary (word presence) | **87.8%** | TF-IDF only reached 66.7% |

TF-IDF underperformed badly for isiXhosa. My hypothesis: isiXhosa builds long words from many prefixes and suffixes, so meaning is spread across many rare word forms, and TF-IDF's frequency weighting handles that poorly. Simple word presence was more robust.

## What I built
- Text cleaning pipeline (lowercasing, punctuation removal, headline + body merging)
- Three feature extractors: bag-of-words, binary presence and TF-IDF
- Multinomial logistic regression in PyTorch with mini-batch SGD and cross-entropy loss
- Two stopping criteria (fixed epochs vs validation-accuracy early stopping)
- Grid search over learning rate and batch size per language
- L1 and L2 regularisation experiments for isiXhosa

## Run it
1. Download the English (`eng`) and isiXhosa (`xho`) splits from MasakhaNEWS into `data/eng/` and `data/xho/` (`train.tsv`, `dev.tsv`, `test.tsv`).
2. `pip install -r requirements.txt`
3. Open `news_classification.ipynb` and run all cells.

*Built for CSC2042S Machine Learning, University of Cape Town (2025).*
