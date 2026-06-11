import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

DATA_PATH = "data/documents.csv"
MODEL_DIR = "models"


def make_training_text(df: pd.DataFrame) -> pd.Series:
    """Combine useful text columns into one text field."""
    text_columns = ["Official name", "Casual name", "Short summary", "Long summary", "Tags"]
    text = pd.Series("", index=df.index, dtype="object")
    for col in text_columns:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
    return text.str.strip()


def get_label_columns(df: pd.DataFrame, min_positive: int = 3) -> list[str]:
    """Select Boolean governance tag columns from documents.csv."""
    useful_prefixes = ("Applications:", "Risk factors:", "Harms:", "Strategies:", "Incentives:")
    extra_labels = ["Primarily applies to the government", "Primarily applies to the private sector"]

    candidate_columns = [col for col in df.columns if col.startswith(useful_prefixes) or col in extra_labels]
    label_columns = []

    for col in candidate_columns:
        positive_count = (df[col] == True).sum()
        negative_count = (df[col] == False).sum()
        if positive_count >= min_positive and negative_count >= min_positive:
            label_columns.append(col)

    return label_columns


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading documents.csv...")
    documents = pd.read_csv(DATA_PATH)

    print("Selecting governance tag columns...")
    label_columns = get_label_columns(documents)
    print(f"Number of selected label columns: {len(label_columns)}")

    print("Creating training text...")
    documents["training_text"] = make_training_text(documents)

    X = documents["training_text"]
    y = documents[label_columns].astype(int)

    print("Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Converting text into TF-IDF features...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=1,  # FIXED: Set to 1 to prevent out-of-vocabulary errors on rare user prompts
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Note: We train this baseline classifier to keep the architecture complete, 
    # but our CLI will use the advanced deep-learning semantic model for tags.
    print("Training baseline multi-label classifier...")
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    classifier.fit(X_train_vec, y_train)

    print("Evaluating baseline model...")
    y_pred = classifier.predict(X_test_vec)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("Saving model files...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "tag_classifier.pkl"))
    joblib.dump(label_columns, os.path.join(MODEL_DIR, "label_columns.pkl"))

    print("Done. Model files saved in the models/ folder.")

if __name__ == "__main__":
    main()