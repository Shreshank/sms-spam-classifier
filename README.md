# SMS Spam Classifier

A machine learning project that detects spam in SMS messages using NLP and multiple classifiers, with a Streamlit web app for live predictions.

---

## Demo

> Enter any SMS message → get an instant spam/ham prediction.

![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)

---

## Dataset

[UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 labeled SMS messages (ham/spam).

| Class | Count |
|-------|-------|
| Ham   | 4,825 |
| Spam  | 747   |

---

## How It Works

```
Raw SMS → Preprocessing → TF-IDF Vectorization → Classifier → Spam / Ham
```

**Preprocessing steps:**
1. Lowercase
2. Tokenize
3. Remove non-alphanumeric tokens
4. Remove stopwords and punctuation
5. Porter Stemming

---

## Models Evaluated

Compared 12 classifiers using 6-fold cross-validation, scoring on accuracy and precision:

| Model | Notes |
|-------|-------|
| MultinomialNB | Strong baseline for text |
| BernoulliNB | Good with binary features |
| ComplementNB | Handles class imbalance well |
| GaussianNB | — |
| KNeighborsClassifier | — |
| LogisticRegression | — |
| SVC (RBF kernel) | — |
| DecisionTree | — |
| AdaBoost | — |
| RandomForest | — |
| XGBoost | — |
| VotingClassifier | RF + MNB + KNN + SVC ensemble |

Final model saved via `joblib` as `model.pkl`.

---

## Project Structure

```
├── spam-detection.ipynb   # EDA, preprocessing, model comparison
├── app.py                 # Streamlit app
├── model.pkl              # Saved classifier
├── text_vector.pkl        # Saved TF-IDF vectorizer
├── spam.csv               # Training dataset
└── dummydata.csv          # Custom test set
```

---

## Setup

```bash
git clone https://github.com/your-username/sms-spam-classifier
cd sms-spam-classifier
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

**requirements.txt**
```
streamlit
scikit-learn
xgboost
nltk
pandas
numpy
joblib
wordcloud
matplotlib
seaborn
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `NLTK` · `TF-IDF` · `Streamlit` · `joblib`
