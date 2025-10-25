import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Optional NLP extras
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except Exception:
    nltk_available = False

# VADER for rule-based sentiment
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    vader_available = True
except Exception:
    vader_available = False

# Transformer (inference only)
try:
    from transformers import pipeline as hf_pipeline
    transformer_available = True
except Exception:
    transformer_available = False

# -------------------------
# Utilities & preprocessing
# -------------------------
def ensure_nltk():
    """Download NLTK data if needed. Returns True if NLTK is available after ensuring."""
    if not nltk_available:
        return False
    try:
        stopwords.words("english")
        nltk.data.find("corpora/wordnet")
    except Exception:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return True

def clean_text(text, lemmatizer=None, stop_words=None):
    """Lowercase, remove urls/html, non-alphanum, collapse spaces, optional stopword removal & lemmatization."""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    # remove urls
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    # remove html tags
    t = re.sub(r"<.*?>", " ", t)
    # keep alphanum and spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    tokens = t.split()
    if stop_words:
        tokens = [w for w in tokens if w not in stop_words]
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def normalize_labels(labels):
    """Map labels to 0/1. Accepts numbers or strings like 'positive','negative'."""
    def map_label(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip().lower()
        if s in ("0", "negative", "neg", "negativ", "negitive", "bad"):
            return 0
        if s in ("1", "positive", "pos", "good"):
            return 1
        # fallback: try numeric parse
        try:
            return int(float(s))
        except Exception:
            # unknown -> treat as negative (0). Change behavior if desired.
            return 0
    return labels.apply(map_label)

# -------------------------
# ML pipeline (TF-IDF + LR)
# -------------------------
def build_ml_pipeline(max_features=20000, ngram_range=(1,2), min_df=2):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    clf = LogisticRegression(solver="saga", max_iter=2000, class_weight="balanced", random_state=42)
    pipeline = Pipeline([("tfidf", vect), ("clf", clf)])
    return pipeline

def train_and_evaluate_ml(df, text_col="text", label_col="label", test_size=0.2, random_state=42):
    # optional NLTK resources
    if ensure_nltk():
        sw = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
    else:
        sw = None
        lem = None

    # clean text
    print("[*] Cleaning texts...")
    df["clean_text"] = df[text_col].astype(str).apply(lambda t: clean_text(t, lem, sw))

    y = normalize_labels(df[label_col])
    X = df["clean_text"]

    # stratify if possible
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    pipeline = build_ml_pipeline()
    print("[*] Training TF-IDF + LogisticRegression...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (rows=true labels, cols=preds):")
    print(confusion_matrix(y_test, preds))
    return pipeline, (X_test, y_test, preds)

# -------------------------
# Rule-based (VADER)
# -------------------------
def ensure_vader():
    if not nltk_available:
        return False
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
    return True

def rule_predict(texts, compound_threshold_pos=0.05, compound_threshold_neg=-0.05):
    """Return 0/1 predictions and scores using VADER. 1 -> positive, 0 -> negative/neutral."""
    if not vader_available:
        # try to initialize after ensuring resource
        if ensure_vader():
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
            except Exception:
                raise RuntimeError("VADER not available. Install NLTK and download vader_lexicon.")
        else:
            raise RuntimeError("NLTK not available for VADER. Install nltk and download vader_lexicon.")
    else:
        sia = SentimentIntensityAnalyzer()

    results = []
    for t in texts:
        s = sia.polarity_scores(str(t))
        comp = s["compound"]
        label = 1 if comp >= compound_threshold_pos else (0 if comp <= compound_threshold_neg else 0)
        results.append((label, comp, s))
    return results

# -------------------------
# Transformer inference
# -------------------------
def transformer_predict(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Use Hugging Face pipeline('sentiment-analysis'). Returns (label, score) pairs.
    Requires `transformers` installed.
    """
    if not transformer_available:
        raise RuntimeError("transformers not installed. pip install transformers torch")
    nlp = hf_pipeline("sentiment-analysis", model=model_name)
    preds = nlp(texts)
    # standardize: label 'POSITIVE'/'NEGATIVE' -> 1/0
    out = []
    for p in preds:
        lab = p.get("label", "")
        score = p.get("score", 0.0)
        mapped = 1 if lab.upper().startswith("POS") else 0
        out.append((mapped, score, p))
    return out

# -------------------------
# Save/Load utilities
# -------------------------
def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    print(f"Saved pipeline to: {path}")

def load_pipeline(path):
    p = joblib.load(path)
    print(f"Loaded pipeline from: {path}")
    return p

def predict_with_pipeline(pipeline, texts):
    """Predict labels and optionally probabilities using a trained sklearn pipeline."""
    cleaned = []
    if ensure_nltk():
        sw = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
    else:
        sw = None
        lem = None
    cleaned = [clean_text(t, lem, sw) for t in texts]
    preds = pipeline.predict(cleaned)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(cleaned)
    return preds, probs

# -------------------------
# CLI and main
# -------------------------
def load_csv(csv_path, text_col="text", label_col="label"):
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{text_col}' and '{label_col}'")
    df = df[[text_col, label_col]].dropna()
    return df

def main(args):
    mode = args.mode.lower()

    if mode == "ml":
        if not args.data:
            print("ML mode requires --data <csv file>", file=sys.stderr)
            sys.exit(2)
        csv_path = Path(args.data)
        if not csv_path.exists():
            print(f"Data file not found: {csv_path}", file=sys.stderr)
            sys.exit(2)
        df = load_csv(csv_path, text_col=args.text_col, label_col=args.label_col)
        pipeline, info = train_and_evaluate_ml(df, text_col=args.text_col, label_col=args.label_col, test_size=args.test_size, random_state=args.random_state)
        if args.model:
            save_pipeline(pipeline, args.model)
        if args.demo:
            samples = [
                "I love this product! It works perfectly and I'm very happy.",
                "This is the worst service I have ever used. Totally disappointed.",
                "The movie was okay, not great but not terrible either."
            ]
            preds, probs = predict_with_pipeline(pipeline, samples)
            print("\nDemo predictions (ml):")
            for i, s in enumerate(samples):
                p = preds[i]
                prob_text = f" (proba: {probs[i].max():.3f})" if probs is not None else ""
                print(f"TEXT: {s}\nPRED: {p}{prob_text}\n---")

    elif mode == "rule":
        # Rule-based using VADER
        if args.demo:
            samples = [
                "Absolutely fantastic experience, would come again!",
                "I hate it. Totally useless and broke immediately.",
                "It was fine. Nothing special."
            ]
        elif args.data:
            df = pd.read_csv(args.data)
            if args.text_col not in df.columns:
                raise ValueError(f"text column '{args.text_col}' not found in CSV.")
            samples = df[args.text_col].astype(str).tolist()
        else:
            print("Rule mode requires --demo or --data", file=sys.stderr)
            sys.exit(2)

        results = rule_predict(samples)
        print("\nRule-based (VADER) results:")
        for i, (lab, comp, full) in enumerate(results[:20]):
            print(f"TEXT: {samples[i]}\nLABEL: {lab}  compound={comp:.3f}  details={full}\n---")

    elif mode == "transformer":
        if not transformer_available:
            print("Transformer mode requires `transformers` package. pip install transformers torch", file=sys.stderr)
            sys.exit(2)
        if args.demo:
            samples = [
                "I absolutely love this! Highly recommended.",
                "Worst purchase ever. Do not buy.",
                "It was okay, a little boring but watchable."
            ]
        elif args.data:
            df = pd.read_csv(args.data)
            if args.text_col not in df.columns:
                raise ValueError(f"text column '{args.text_col}' not found in CSV.")
            samples = df[args.text_col].astype(str).tolist()
        else:
            print("Transformer mode requires --demo or --data", file=sys.stderr)
            sys.exit(2)
        out = transformer_predict(samples)
        print("\nTransformer results:")
        for i, (lab, score, raw) in enumerate(out[:20]):
            print(f"TEXT: {samples[i]}\nLABEL: {lab}  score={score:.4f}  raw={raw}\n---")

    else:
        print("Unknown mode. Choose from: ml, rule, transformer", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis script with ML, rule-based, and transformer options.")
    parser.add_argument("--mode", type=str, default="ml", help="mode: ml | rule | transformer")
    parser.add_argument("--data", type=str, help="CSV file path (required for ml training).")
    parser.add_argument("--text-col", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label-col", type=str, default="label", help="Name of the label column")
    parser.add_argument("--model", type=str, default="sentiment_pipeline.joblib", help="Output path for saved sklearn pipeline")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction for ML mode")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions/examples")
    args = parser.parse_args()
    main(args)
