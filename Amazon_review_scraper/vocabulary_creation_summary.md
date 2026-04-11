# Vocabulary Creation Notebook — Line-by-Line Summary

> This file explains every single line of code in `vocabulary_creation.ipynb`
> in plain English so that anyone (even a beginner) can understand what is happening and why.

---

## WHAT THIS NOTEBOOK DOES (Big Picture)

We have a CSV file with 100 Amazon phone reviews. The notebook:
1. **Reads** and **cleans** the review text
2. **Builds a vocabulary** (list of all unique words) — both manually and using sklearn
3. **Converts text into numbers** using 3 methods: One-Hot Encoding, Bag of Words, TF-IDF
4. **Compares** these 3 methods and explains which is better and why
5. **Analyzes sparsity** — why most of these number-matrices are full of zeros and why that's a problem

---

## Section 0: Install Dependencies

```python
%pip install pandas scikit-learn matplotlib --quiet
```
- `%pip install` → Jupyter magic command that installs Python packages directly from the notebook
- `pandas` → library for reading CSV files and working with tables
- `scikit-learn` → machine learning library (we use it for CountVectorizer and TfidfVectorizer)
- `matplotlib` → library for drawing charts and graphs
- `--quiet` → don't print noisy installation logs, just install silently

---

## Section 1: Import Required Libraries

```python
import pandas as pd
```
- Loads the `pandas` library and gives it a short nickname `pd`
- pandas is used to read CSV files and work with data in table format (DataFrames)

```python
import re
```
- Loads Python's built-in `re` (Regular Expressions) module
- We use it to find and remove unwanted characters (punctuation, digits) from text

```python
import string
```
- Loads Python's built-in `string` module (provides constants like `string.punctuation`)

```python
from collections import Counter
```
- Imports `Counter` from Python's built-in `collections` module
- Counter is like a dictionary that automatically counts how many times each item appears
- Example: `Counter(["a", "b", "a"])` → `{"a": 2, "b": 1}`

```python
from sklearn.feature_extraction.text import CountVectorizer
```
- Imports `CountVectorizer` from scikit-learn
- CountVectorizer takes a list of text documents and converts them into a matrix of word counts
- It's the "automatic" way to build vocabulary + Bag of Words

```python
import matplotlib.pyplot as plt
```
- Loads `matplotlib.pyplot` (nicknamed `plt`) for creating charts/plots

```python
print("Libraries imported successfully.")
```
- Just prints a confirmation message so you know nothing crashed

---

## Section 2: Load CSV File

```python
df = pd.read_csv("amazon_reviews_preprocessed_clean.csv")
```
- `pd.read_csv(...)` → reads the CSV file and loads it into a DataFrame (a table with rows and columns)
- `df` → the variable name for our DataFrame (short for "dataframe")

```python
print(f"Shape: {df.shape}")
```
- `df.shape` → returns (number_of_rows, number_of_columns), e.g. `(100, 5)`
- This tells us: we have 100 reviews and 5 columns

```python
print(f"Columns: {list(df.columns)}\n")
```
- `df.columns` → gives the names of all columns (e.g. review_text, rating, cleaned_text, etc.)

```python
df.head(3)
```
- Shows the first 3 rows of the table so we can visually inspect the data

---

## Section 3: Text Preprocessing

```python
texts = df["cleaned_text"].dropna().astype(str).tolist()
```
- `df["cleaned_text"]` → picks just the "cleaned_text" column from the table
- `.dropna()` → removes any rows where cleaned_text is empty/missing (NaN)
- `.astype(str)` → makes sure every value is a string (text), not a number
- `.tolist()` → converts from a pandas Series to a plain Python list
- Result: `texts` is a list of 100 text strings

```python
def preprocess(text):
```
- Defines a function called `preprocess` that takes one piece of text and cleans it

```python
    text = text.lower()
```
- Converts the entire text to lowercase
- Example: `"Phone is GREAT"` → `"phone is great"`
- Why? So "Phone" and "phone" are treated as the same word

```python
    text = re.sub(r"[^a-z\s]", " ", text)
```
- `re.sub` → "substitute" (find & replace using a pattern)
- `r"[^a-z\s]"` → matches any character that is NOT a lowercase letter (a-z) or whitespace (\s)
- Replaces all punctuation, digits, special characters with a space
- Example: `"phone's 5g!"` → `"phone s  g "`

```python
    text = re.sub(r"\s+", " ", text).strip()
```
- `r"\s+"` → matches one or more consecutive spaces
- Replaces multiple spaces with a single space
- `.strip()` → removes leading/trailing spaces
- Example: `"phone s  g "` → `"phone s g"`

```python
    return text
```
- Returns the cleaned text back to whoever called this function

```python
cleaned_texts = [preprocess(t) for t in texts]
```
- This is a "list comprehension" — a shortcut to apply `preprocess()` to every review
- It's the same as writing a for loop but in one line
- Result: `cleaned_texts` is a list of 100 cleaned text strings

```python
tokenized = [t.split() for t in cleaned_texts]
```
- `.split()` → splits a string into a list of words by spaces
- Example: `"phone is great"` → `["phone", "is", "great"]`
- Result: `tokenized` is a list of 100 lists, where each inner list contains individual words

```python
print(f"Total reviews: {len(cleaned_texts)}")
```
- Prints how many reviews we have (100)

```python
print(f"Sample tokens (review 0): {tokenized[0][:20]}")
```
- Shows the first 20 words from the first review so we can verify tokenization worked

---

## Section 4: Build Vocabulary Manually

```python
all_tokens = [token for tokens in tokenized for token in tokens]
```
- **Flattens** the list of lists into one big list of ALL words
- `tokenized` = `[["phone", "is"], ["good", "phone"]]`
- `all_tokens` = `["phone", "is", "good", "phone"]`
- This is a nested list comprehension: outer loop goes through each document, inner loop goes through each word

```python
word_freq = Counter(all_tokens)
```
- Counts how many times each word appears across ALL reviews
- Result: `{"phone": 236, "camera": 203, "good": 200, ...}`

```python
manual_vocab = sorted(word_freq.keys())
```
- `word_freq.keys()` → gets just the unique words (no counts)
- `sorted(...)` → arranges them in alphabetical order
- Result: `["acceptable", "ai", "allround", ..., "worth"]` — our vocabulary!

```python
manual_word2idx = {word: idx for idx, word in enumerate(manual_vocab)}
```
- Creates a dictionary mapping each word to a number (its position in the vocabulary)
- Example: `{"acceptable": 0, "ai": 1, "allround": 2, ...}`
- This is essential for building OHE/BOW vectors later — we need to know which column each word belongs to

```python
print(f"Manual Vocabulary Size : {len(manual_vocab)}")
```
- Prints how many unique words we found (282)

```python
print(f"Total Tokens (with repeats) : {len(all_tokens)}")
```
- Prints total word count including repeats (8638)

```python
top20_manual = word_freq.most_common(20)
```
- Gets the 20 most frequent words and their counts
- Returns a list of tuples: `[("phone", 236), ("camera", 203), ...]`

```python
for rank, (word, freq) in enumerate(top20_manual, 1):
    print(f"  {rank:>2}. {word:<20} {freq}")
```
- Loops through the top 20 words and prints each with its rank and frequency
- `enumerate(..., 1)` → starts counting from 1 instead of 0
- `{rank:>2}` → right-align the number in 2 spaces
- `{word:<20}` → left-align the word in 20 spaces (for neat columns)

---

## Section 5: Build Vocabulary Using Sklearn

```python
vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-z]{2,}\b")
```
- Creates a CountVectorizer object — sklearn's automatic vocabulary builder
- `token_pattern=r"(?u)\b[a-z]{2,}\b"` → tells it to only consider words that:
  - Are lowercase letters only
  - Are at least 2 characters long (ignores single letters like "a", "i")
- `(?u)` = Unicode-aware, `\b` = word boundary, `{2,}` = 2 or more characters

```python
X = vectorizer.fit_transform(cleaned_texts)
```
- `fit_transform` does TWO things in one call:
  1. **fit** → learns the vocabulary from all 100 reviews
  2. **transform** → converts each review into a word-count vector
- `X` is a sparse matrix of shape (100, 276) — 100 documents × 276 unique words
- Each cell = how many times that word appears in that document

```python
sklearn_vocab = vectorizer.vocabulary_
```
- Gets the learned vocabulary as a dictionary: `{"phone": 200, "camera": 45, ...}`
- The numbers are column indices (which column in matrix X each word maps to)

```python
sklearn_vocab_list = sorted(sklearn_vocab.keys())
```
- Gets just the words, sorted alphabetically

```python
word_totals = X.sum(axis=0).A1
```
- `X.sum(axis=0)` → sums each column (= total count of each word across all documents)
- `.A1` → converts from a matrix to a flat 1D array
- Result: an array like `[5, 12, 200, ...]` — total count of each word

```python
sklearn_freq = {w: int(word_totals[sklearn_vocab[w]]) for w in sklearn_vocab_list}
```
- Creates a dictionary `{word: total_count}` for every word in the sklearn vocabulary
- `sklearn_vocab[w]` → gets the column index for word `w`
- `word_totals[...]` → looks up the total count at that column index

```python
top20_sklearn = sorted(sklearn_freq.items(), key=lambda x: x[1], reverse=True)[:20]
```
- Sorts words by their count (highest first) and takes the top 20
- `key=lambda x: x[1]` → sort by the second element (the count), not the word
- `reverse=True` → descending order (biggest first)
- `[:20]` → take only the first 20

---

## Section 6: Store Vocabulary Variables

This cell just prints a summary of all the variables we've created so far.
These variables will be reused in later sections for OHE, BOW, and TF-IDF.

```python
for w in list(manual_word2idx)[:10]:
    print(f"    '{w}' -> {manual_word2idx[w]}")
```
- Shows the first 10 word→index mappings as a sample

---

## Section 7: Visualize Top Frequent Words

```python
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
```
- Creates a figure with 2 side-by-side subplots (1 row, 2 columns)
- `figsize=(18, 6)` → makes it 18 inches wide, 6 inches tall

```python
m_words  = [w for w, _ in top20_manual]
m_counts = [c for _, c in top20_manual]
```
- Extracts words and counts from the top-20 list into separate lists
- `_` means "I don't care about this value" — we only want one of the two

```python
axes[0].barh(m_words[::-1], m_counts[::-1], color="steelblue")
```
- `barh` → draws a **horizontal** bar chart
- `[::-1]` → reverses the list so the highest-frequency word is at the top
- `color="steelblue"` → sets bar color

```python
for i, v in enumerate(m_counts[::-1]):
    axes[0].text(v + 0.3, i, str(v), va="center", fontsize=9)
```
- Adds the count number label at the end of each bar

### Zipf's Law Plot

```python
all_freqs = sorted(word_freq.values(), reverse=True)
```
- Gets all word frequencies sorted from most common to least common

```python
plt.yscale("log")
plt.xscale("log")
```
- Sets both axes to logarithmic scale
- On a log-log plot, Zipf's Law appears as a straight line
- Zipf's Law says: the 2nd most common word appears ~half as often as the 1st, the 3rd ~1/3 as often, etc.

---

## Section 8: One-Hot Encoding (OHE)

**What is OHE?** Each document becomes a row of 0s and 1s. If a word exists in the document → 1, otherwise → 0. It does NOT count how many times a word appears.

```python
import numpy as np
```
- Imports NumPy — the library for working with number arrays/matrices

```python
vocab_size = len(manual_vocab)
```
- Stores the vocabulary size (282) — this will be the number of columns

```python
ohe_matrix = np.zeros((len(tokenized), vocab_size), dtype=int)
```
- Creates a matrix filled with zeros
- Shape: (100 rows × 282 columns) — one row per document, one column per word
- `dtype=int` → stores integers (0 and 1)

```python
for doc_idx, tokens in enumerate(tokenized):
```
- Loops through each document (doc_idx = 0, 1, 2, ... 99)
- `tokens` = the list of words in that document

```python
    unique_tokens = set(tokens)
```
- `set(...)` → removes duplicate words (we only care IF a word exists, not how many times)

```python
    for token in unique_tokens:
        if token in manual_word2idx:
            ohe_matrix[doc_idx, manual_word2idx[token]] = 1
```
- For each unique word in this document:
  - Look up which column it belongs to (`manual_word2idx[token]`)
  - Set that cell to 1

```python
ohe_df = pd.DataFrame(ohe_matrix, columns=manual_vocab)
```
- Wraps the NumPy matrix in a pandas DataFrame with column names = vocabulary words
- Makes it easier to view and work with

### OHE Verification Cell

```python
print(f"Unique values in OHE matrix: {np.unique(ohe_matrix)}")
```
- Confirms the matrix only contains [0, 1] — no other values

```python
print(f"Sparsity: {(1 - ohe_matrix.mean()) * 100:.1f}%")
```
- Calculates what percentage of the matrix is zeros
- `ohe_matrix.mean()` → average value (close to 0 because mostly zeros)
- `1 - mean` → fraction of zeros
- `* 100` → convert to percentage

---

## Section 9: Bag of Words (BOW)

**What is BOW?** Like OHE but instead of just 0/1, it stores the ACTUAL COUNT of how many times each word appears in each document.

```python
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
```
- `X` was the sparse matrix created by CountVectorizer back in Section 5
- `.toarray()` → converts sparse matrix to a regular dense array (so we can see all values)
- `get_feature_names_out()` → gets the list of vocabulary words (column names)

### BOW vs OHE Comparison Cell

```python
bow_row = X[0].toarray().flatten()
```
- Gets the BOW vector for document 0
- `.flatten()` → converts from a 2D (1×276) matrix to a 1D (276,) array

```python
multi_count = [(feature_names[i], bow_row[i]) for i in range(len(bow_row)) if bow_row[i] > 1]
```
- Finds all words that appear MORE than once in document 0
- These are the words where BOW gives different values than OHE
  - OHE would say 1 (present), BOW says the actual count (2, 3, 4, etc.)

---

## Section 10: TF-IDF

**What is TF-IDF?** It stands for "Term Frequency × Inverse Document Frequency."
- **TF** = how often a word appears in THIS document
- **IDF** = how RARE the word is across ALL documents
- **TF-IDF** = TF × IDF → gives HIGH score to words that are frequent in one doc but rare overall

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- Imports sklearn's TF-IDF tool

```python
tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-z]{2,}\b")
```
- Creates a TfidfVectorizer with the same token pattern as CountVectorizer
- Only considers words with 2+ lowercase letters

```python
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_texts)
```
- Learns vocabulary AND computes TF-IDF scores for all documents in one step
- Result: a matrix of float values (not integers like BOW, not 0/1 like OHE)

```python
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
```
- Gets the list of vocabulary words

```python
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)
```
- Converts to a readable DataFrame

### Top TF-IDF Words per Document

```python
row = tfidf_matrix[doc_idx].toarray().flatten()
```
- Gets the TF-IDF vector for one document

```python
top_indices = row.argsort()[-5:][::-1]
```
- `argsort()` → returns indices that would sort the array (lowest to highest)
- `[-5:]` → takes the last 5 (the 5 highest values)
- `[::-1]` → reverses to get highest first
- Result: indices of the 5 words with the highest TF-IDF score in this document

```python
mean_tfidf = tfidf_matrix.mean(axis=0).A1
```
- Calculates the average TF-IDF score for each word across ALL documents
- This tells us which words are generally important vs. generally unimportant

```python
word_importance = sorted(zip(tfidf_feature_names, mean_tfidf), key=lambda x: x[1], reverse=True)
```
- `zip(...)` → pairs each word with its average TF-IDF score
- Sorts by score (highest first)
- Top words = most distinctive; bottom words = most common/generic

---

## Section 11: Comparison Table (OHE vs BoW vs TF-IDF)

```python
sample_words = ["phone", "camera", "good", "battery", "oneplus",
                "waterproof", "nitpick", "blazing", "verdict", "premium"]
```
- Picks 10 words — a mix of common words (phone, camera) and rare words (waterproof, nitpick)
- This lets us see how the 3 methods treat common vs. rare words differently

```python
sample_words = [w for w in sample_words if w in tfidf_df.columns]
```
- Filters out any words that aren't in our vocabulary (safety check)

```python
doc_freq = (tfidf_matrix.toarray() > 0).sum(axis=0)
```
- For each word, counts in how many documents it appears
- `> 0` → converts to True/False (is the TF-IDF score positive?)
- `.sum(axis=0)` → counts the Trues for each column (each word)

```python
idf_values = tfidf_vectorizer.idf_
```
- Gets the IDF (Inverse Document Frequency) score for each word
- High IDF = rare word, Low IDF = common word

```python
comparison_table = pd.DataFrame(comparison_rows)
```
- Creates a nice table showing: Word, OHE value, BoW value, TF-IDF value, Doc Frequency, IDF

---

## Section 12: Why Common Words Get Lower TF-IDF Weight

```python
word_doc_freq = list(zip(tfidf_feature_names, doc_freq, idf_values))
word_doc_freq.sort(key=lambda x: x[1], reverse=True)
```
- Packages each word with its document frequency and IDF value
- Sorts by document frequency (most common first)

This cell prints two tables:
1. **MOST COMMON words** → appear in many documents → LOW IDF → LOW TF-IDF
2. **MOST RARE words** → appear in few documents → HIGH IDF → HIGH TF-IDF

### IDF vs Document Frequency Scatter Plot

```python
axes[0].scatter(doc_freq, idf_values, alpha=0.6, color="teal", edgecolors="k", s=40)
```
- Draws a scatter plot where:
  - X-axis = document frequency (in how many docs the word appears)
  - Y-axis = IDF value
- Shows the inverse relationship: as doc frequency goes UP, IDF goes DOWN

### Common vs Rare Bar Chart

```python
colors = ["#e74c3c"] * 10 + ["#2ecc71"] * 10
```
- Red bars for the 10 most common words, green bars for the 10 rarest words
- Visually shows that rare words (green) have higher average TF-IDF scores

---

## Section 13: Sparse Matrix Analysis

### Matrix Shapes

```python
matrices = {
    "One-Hot Encoding (OHE)": ohe_df.values,
    "Bag of Words (BoW)":     bow_df.values,
    "TF-IDF":                 tfidf_df.values,
}
```
- Creates a dictionary of all 3 matrices for easy looping
- `.values` → extracts the raw NumPy array from each DataFrame

```python
total = mat.shape[0] * mat.shape[1]
```
- Calculates total number of elements (cells) in the matrix
- For a 100×276 matrix, that's 27,600 cells

### Sparsity Calculation

```python
zeros = np.count_nonzero(mat == 0)
```
- `mat == 0` → creates a True/False matrix (True where value is 0)
- `np.count_nonzero(...)` → counts how many Trues (= how many zeros)

```python
sparsity = (zeros / total) * 100
```
- Sparsity = percentage of the matrix that is zeros
- Example: if 95% of cells are zero, sparsity = 95%

### Dense vs Sparse Memory Comparison

```python
dense_bytes = mat.nbytes
```
- `.nbytes` → how many bytes the matrix uses in memory as a dense array
- Every cell takes 8 bytes (float64) regardless of whether it's zero

```python
sparse_mat = sparse.csr_matrix(mat)
```
- Converts to CSR (Compressed Sparse Row) format
- CSR only stores non-zero values and their positions — skips all the zeros

```python
sparse_bytes = (sparse_mat.data.nbytes + sparse_mat.indices.nbytes + sparse_mat.indptr.nbytes)
```
- Calculates total memory used by the sparse format:
  - `.data` → the actual non-zero values
  - `.indices` → which column each non-zero value is in
  - `.indptr` → where each row starts in the data array

### Sparsity Heatmap

```python
subset = (mat[:20, :50] != 0).astype(int)
ax.imshow(subset, cmap="YlOrRd", aspect="auto", interpolation="nearest")
```
- Takes a small slice (20 docs × 50 words) of each matrix
- Converts to 0/1 (is the value non-zero?)
- Shows as a heatmap: yellow = zero, red = non-zero
- Visually demonstrates how sparse (mostly yellow) the matrices are

### Scale-up Simulation

```python
scenarios = [
    ("Our dataset",           100,       276),
    ("Medium corpus",      10_000,    50_000),
    ("Large corpus",      100_000,   200_000),
    ("Web-scale corpus", 1_000_000,  500_000),
]
```
- Simulates what would happen if we had bigger datasets
- Shows that at web scale (1M docs × 500K words), a dense matrix needs ~3.7 TB of RAM!

```python
dense_gb  = (total_elements * 8) / (1024**3)
```
- Each element = 8 bytes (float64)
- Divides by 1024³ to convert bytes → gigabytes

```python
sparse_gb = (total_elements * 0.01 * 12) / (1024**3)
```
- Assumes 99% sparsity (only 1% non-zero)
- Each non-zero element needs ~12 bytes (value + column index + overhead)

---

## KEY TAKEAWAYS

| What We Built | Purpose |
|---|---|
| `manual_vocab` | List of 282 unique words found manually |
| `manual_word2idx` | Maps each word to a number (index) |
| `sklearn_vocab` | Same thing but built automatically by sklearn (276 words — slightly fewer because it filters 1-char words) |
| `ohe_df` | One-Hot Encoding: 0/1 matrix (is word present?) |
| `bow_df` | Bag of Words: count matrix (how many times does word appear?) |
| `tfidf_df` | TF-IDF: weighted matrix (how important is the word?) |

### Which method is best?
- **OHE** → Simplest but loses frequency info. Good for quick baselines.
- **BoW** → Captures frequency but treats "phone" (appears everywhere) same as "waterproof" (rare and meaningful).
- **TF-IDF** → Best of the three. Automatically down-weights common words and highlights distinctive words.
- **All three are sparse** → mostly zeros, wasteful at scale. Modern NLP uses dense embeddings (Word2Vec, BERT) instead.
