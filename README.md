# NLP Text Feature Engineering — Amazon Product Reviews

## Project Overview

This project demonstrates a complete **Natural Language Processing (NLP) pipeline** — from scraping real Amazon product reviews to converting raw text into numerical feature vectors that machine learning models can understand.

---

## Workflow

```
Amazon Website → Scrape Reviews → Clean & Preprocess → Build Vocabulary → Feature Engineering → Analysis
```

### Step 1: Data Collection (Web Scraping)
- Navigated to [Amazon.in](https://www.amazon.in)
- Searched for a product (OnePlus Nord 5)
- Extracted **100 customer reviews** including review text, rating, reviewer name, and review date
- Cleaned and preprocessed the scraped text (lowercased, removed stopwords, special characters, etc.)
- Saved the final cleaned data as **`amazon_reviews_preprocessed_clean.csv`**

### Step 2: Vocabulary Creation & Feature Engineering (This Notebook)
- Built vocabulary manually and using sklearn
- Converted text into numerical representations using **3 methods**:
  1. One-Hot Encoding (OHE)
  2. Bag of Words (BoW)
  3. TF-IDF
- Compared all 3 methods and analyzed sparsity

---

## Files

| File | Description |
|---|---|
| `amazon_reviews_preprocessed_clean.csv` | Cleaned Amazon reviews dataset (100 reviews, 5 columns) |
| `vocabulary_creation.ipynb` | Main Jupyter notebook with all code and outputs |
| `vocabulary_creation_summary.md` | Line-by-line code explanation for beginners |

---

## Dataset Columns

| Column | Description |
|---|---|
| `review_text` | Original review text as scraped from Amazon |
| `rating` | Star rating given by the reviewer |
| `reviewer_name` | Name of the reviewer |
| `review_date` | Date the review was posted |
| `cleaned_text` | Preprocessed text (lowercased, stopwords removed, no punctuation) |

---

## Notebook Sections

| # | Section | What It Does |
|---|---|---|
| 0 | Install Dependencies | Installs pandas, scikit-learn, matplotlib |
| 1 | Import Libraries | Loads all required Python libraries |
| 2 | Load CSV | Reads the cleaned reviews CSV into a DataFrame |
| 3 | Text Preprocessing | Additional cleaning — lowercase, remove digits/punctuation, tokenize |
| 4 | Build Vocabulary (Manual) | Uses `collections.Counter` to count words and build a sorted vocabulary |
| 5 | Build Vocabulary (Sklearn) | Uses `CountVectorizer` to automatically build vocabulary + document-term matrix |
| 6 | Store Variables | Saves vocabulary and word-to-index mappings for reuse |
| 7 | Visualize Frequent Words | Bar charts of top-20 words + Zipf's Law frequency distribution plot |
| 8 | One-Hot Encoding | Binary (0/1) document vectors — marks word presence only |
| 9 | Bag of Words | Integer count vectors — counts how many times each word appears |
| 10 | TF-IDF | Weighted vectors — highlights important words, suppresses common ones |
| 11 | Comparison Table | Side-by-side comparison of OHE vs BoW vs TF-IDF with real examples |
| 12 | TF-IDF Weight Analysis | Explains why common words get lower TF-IDF weight (IDF formula) |
| 13 | Sparse Matrix Analysis | Sparsity %, memory comparison, heatmaps, scale-up simulation |

---

## Key Concepts Covered

- **Vocabulary Creation** — Building a word-to-index mapping from raw text
- **One-Hot Encoding (OHE)** — Binary presence/absence vectors
- **Bag of Words (BoW)** — Word frequency count vectors
- **TF-IDF** — Term Frequency × Inverse Document Frequency weighted vectors
- **Sparsity** — Why >90% zeros in these matrices is a problem at scale
- **Zipf's Law** — Natural frequency distribution of words in any language
- **IDF** — Why words appearing in every document are less informative

---

## Key Results

| Metric | Value |
|---|---|
| Total Reviews | 100 |
| Manual Vocabulary Size | 282 unique words |
| Sklearn Vocabulary Size | 276 unique words |
| Most Frequent Word | "phone" (236 occurrences) |
| Matrix Sparsity (OHE) | ~70%+ zeros |
| Matrix Sparsity (BoW) | ~70%+ zeros |
| Matrix Sparsity (TF-IDF) | ~70%+ zeros |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Programming language |
| pandas | Data loading and manipulation |
| scikit-learn | CountVectorizer, TfidfVectorizer |
| matplotlib | Charts and visualizations |
| NumPy | Matrix operations |
| SciPy | Sparse matrix analysis |
| Jupyter Notebook | Interactive development environment |
| collections.Counter | Manual word frequency counting |

---

## How to Run

### Prerequisites
- Python 3.8+ (Anaconda recommended)
- Jupyter Notebook or VS Code with Jupyter extension

### Steps

1. **Clone or download** this project folder

2. **Open a terminal** in the project directory

3. **Install dependencies** (if not using Anaconda):
   ```bash
   pip install pandas scikit-learn matplotlib scipy jupyter
   ```

4. **Launch the notebook**:
   ```bash
   jupyter notebook vocabulary_creation.ipynb
   ```
   Or open in VS Code and select the **Python 3 (Anaconda)** kernel.

5. **Run All Cells** — the notebook is self-contained and runs top to bottom.

> **Note:** The CSV file (`amazon_reviews_preprocessed_clean.csv`) must be in the same directory as the notebook.

---

## What's Next?

These sparse vector representations (OHE, BoW, TF-IDF) are the **classical approach** to text feature engineering. Modern NLP has moved to **dense embeddings** that solve the sparsity and semantics problems:

| Classical (This Project) | Modern Alternative |
|---|---|
| OHE / BoW / TF-IDF | Word2Vec, GloVe, FastText |
| Sparse, high-dimensional (276+ dims) | Dense, compact (100–300 dims) |
| No semantic understanding | Captures meaning ("good" ≈ "great") |
| Vocabulary-dependent | Pre-trained on billions of words |
| — | BERT, GPT (contextual embeddings) |

---

## Author

Arya Abinash

---

## License

This project is for educational and learning purposes.

