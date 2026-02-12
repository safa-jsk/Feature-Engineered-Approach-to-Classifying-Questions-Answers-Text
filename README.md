# Feature-Engineered Approach to Classifying Question–Answer Text

> Course project for **CSE440: Natural Language Processing** (BRAC University)

This project compares traditional sparse text features (**TF-IDF**) and dense pretrained embeddings (**GloVe 100d**) for **10-class QA topic classification**.  
Two text variants are evaluated:

1. **Standard Data**: cleaned question-answer text after HTML removal  
2. **Feature-Engineered Data**: combined **Question Title + Question Content + Best Answer** with engineered token phrases

Models include Logistic Regression, feed-forward DNNs, and recurrent architectures (RNN, GRU, LSTM, plus bidirectional variants), with base and hyperparameter-tuned versions.

---

## Authors

- Mohammad Jabir Safa Khandoker  
- Md. Saadat Rahman  
- Nafiur Rahman Afnan  

Department of CSE, BRAC University, Dhaka, Bangladesh

---

## Abstract

We evaluate TF-IDF and GloVe representations for multiclass text classification on a balanced QA dataset spanning 10 domains.  
After preprocessing (HTML stripping, lowercasing, tokenization, stopword removal, lemmatization), we train classical and neural models and compare them using **Accuracy**, **Macro F1**, and **AUC** (where applicable).

---

## Dataset

**Question Answer Classification Dataset** with **10 classes**:

1. Science & Mathematics  
2. Business & Finance  
3. Society & Culture  
4. Sports  
5. Education & Reference  
6. Health  
7. Entertainment & Music  
8. Computers & Internet  
9. Politics & Government  
10. Family & Relationships

### Split

- **Train:** 93,333 samples  
- **Test:** 59,999 samples  
- Balanced class distribution across categories

Each sample contains structured QA text with HTML formatting:
- Question Title
- Question Content
- Best Answer

---

## Preprocessing Pipeline

### 1) HTML Cleaning
- Detect/remove HTML tags (`<[^>]+>`) while preserving text

### 2) Feature Engineering
- Extract and combine:
  - Question Title
  - Question Content
  - Best Answer
- Build both standard and feature-engineered variants for comparison

### 3) NLTK Normalization
- Lowercasing
- URL/email removal
- Non-alphabetic filtering
- Tokenization
- Stopword removal
- Lemmatization

### 4) Label Setup
- LabelEncoder for class IDs (`0..9`)
- One-hot encoding for neural networks

---

## Representations & Model Configurations

## A) TF-IDF Setup

- `dtype=np.float32`
- `max_df=0.9`
- `min_df=5`
- `max_features=30000`
- `ngram_range=(1,2)`
- `stop_words='english'`

### Models
- Logistic Regression (base + tuned)
- DNN on TF-IDF (base + tuned)

---

## B) GloVe Setup

- Pretrained embedding: `glove.6B.100d.txt`
- Embedding dimension: **100**
- Sequence length strategy:
  - Standard mean length: 78
  - Feature-engineered mean length: 70
  - Unified length used: **74**
- Shared neural training setup:
  - Optimizer: Adam (`lr=0.001`)
  - Loss: Categorical Cross-Entropy
  - Metric: Accuracy
  - Batch size: 32
  - Epochs: up to 1000 with early stopping (`patience=10`, restore best weights)

### Neural Architectures
- Embedded DNN
- Simple RNN
- GRU
- LSTM
- Bidirectional RNN
- Bidirectional GRU
- Bidirectional LSTM  
(all with base + hyperparameter-tuned variants)

---

## Results Summary

## Best Overall Model
**GloVe + BiLSTM (Hyperparameter-Tuned) on Feature-Engineered Data**
- **Accuracy:** `0.7017`
- **Macro F1:** `0.6992`
- **AUC:** `0.9493`

## Best Classical ML Model
**TF-IDF + Logistic Regression (Base) on Standard Data**
- **Accuracy:** `0.6834`
- **Macro F1:** `0.6817`
- **AUC:** `0.9383`

## Worst Overall Model
**GloVe + RNN (Hyperparameter-Tuned) on Standard Data**
- **Accuracy:** `0.4245`
- **Macro F1:** `0.3959`
- **AUC:** `0.8237`

---

## Full Performance Table (from report)

| Model | Standard Acc. | Standard F1 | Standard AUC | FE Acc. | FE F1 | FE AUC |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 0.6834 | 0.6817 | 0.9383 | 0.6830 | 0.6814 | 0.9380 |
| TF-IDF + LR (HPT) | 0.6751 | 0.6725 | 0.9339 | 0.6742 | 0.6716 | 0.9339 |
| TF-IDF + DNN | 0.6593 | 0.6527 | — | 0.6577 | 0.6526 | — |
| TF-IDF + DNN (HPT) | 0.6549 | 0.6496 | — | 0.6571 | 0.6524 | — |
| GloVe + DNN | 0.5710 | 0.5632 | 0.8994 | 0.5815 | 0.5748 | 0.9049 |
| GloVe + DNN (HPT) | 0.5751 | 0.5704 | 0.9022 | 0.5839 | 0.5774 | 0.9053 |
| GloVe + RNN | 0.5129 | 0.4983 | 0.8693 | 0.5103 | 0.5001 | 0.8627 |
| GloVe + RNN (HPT) | 0.4245 | 0.3959 | 0.8237 | 0.4608 | 0.4386 | 0.8416 |
| GloVe + GRU | 0.6987 | 0.6934 | 0.9485 | 0.6986 | 0.6957 | 0.9480 |
| GloVe + GRU (HPT) | 0.7037 | 0.6969 | 0.9489 | 0.7000 | 0.6945 | 0.9482 |
| GloVe + LSTM | 0.6963 | 0.6920 | 0.9460 | 0.6952 | 0.6907 | 0.9461 |
| GloVe + LSTM (HPT) | 0.7003 | 0.6956 | 0.9477 | 0.7033 | 0.6984 | 0.9481 |
| GloVe + BiRNN | 0.6393 | 0.6345 | 0.9190 | 0.6372 | 0.6340 | 0.9192 |
| GloVe + BiRNN (HPT) | 0.6145 | 0.6059 | 0.9050 | 0.6343 | 0.6277 | 0.9159 |
| GloVe + BiGRU | 0.6956 | 0.6901 | 0.9473 | 0.6963 | 0.6916 | 0.9477 |
| GloVe + BiGRU (HPT) | 0.7009 | 0.6944 | 0.9483 | 0.7012 | 0.6940 | 0.9487 |
| GloVe + BiLSTM | 0.6970 | 0.6922 | 0.9479 | 0.6985 | 0.6935 | 0.9481 |
| GloVe + BiLSTM (HPT) | 0.7005 | 0.6980 | 0.9488 | 0.7017 | 0.6992 | 0.9493 |

> FE = Feature-Engineered data, HPT = Hyperparameter-Tuned

---

## Key Takeaways

- Gated architectures (**GRU/LSTM**) consistently outperform simple RNN baselines.
- Bidirectional recurrent models improve contextual capture for multiclass QA categorization.
- Feature engineering (Title + Content + Best Answer) provides measurable gains, especially for deep models.
- A strong classical baseline (**TF-IDF + LR**) remains competitive and efficient.

---

## Reproducibility Notes

If you are creating this as a GitHub repo, add your actual code/notebooks and keep this structure:

```text
.
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── reports/
│   └── 03_22201108_22201101_24141074.pdf
└── README.md
```

Suggested `requirements.txt` (adjust to your exact versions):

```txt
numpy
pandas
scikit-learn
tensorflow
keras
nltk
matplotlib
seaborn
wordcloud
```

---

## Acknowledgment

Special thanks for computational support (including NVIDIA RTX 4070 access) and development-phase technical guidance as acknowledged in the report.

---

## References

1. C. C. Aggarwal and C. Zhai, *A Survey of Text Classification Algorithms*, in *Mining Text Data*, Springer, 2012.  
2. J. Pennington, R. Socher, C. Manning, *GloVe: Global Vectors for Word Representation*, EMNLP, 2014.  
3. S. Hochreiter, J. Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997.  
4. K. Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder for SMT*, arXiv:1406.1078, 2014.
