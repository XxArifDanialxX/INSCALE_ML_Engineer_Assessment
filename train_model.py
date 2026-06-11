# Import the tools we need to manage files, handle data, and do machine learning math
import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Tell the code where to find the data and where to save the finished models
DATA_PATH = "data/documents.csv"
MODEL_DIR = "models"


def make_training_text(df: pd.DataFrame) -> pd.Series:
    # This function takes all the separate text parts (like titles and summaries)
    # and squishes them together into one giant paragraph. 
    # This makes it easier for the computer to read everything at once.
    text_columns = ["Official name", "Casual name", "Short summary", "Long summary", "Tags"]
    text = pd.Series("", index=df.index, dtype="object")
    
    for col in text_columns:
        if col in df.columns:
            # Add a space and glue the next column's text to the giant paragraph
            text = text + " " + df[col].fillna("").astype(str)
            
    return text.str.strip()


def get_label_columns(df: pd.DataFrame, min_positive: int = 3) -> list[str]:
    # This function looks at the spreadsheet to find the legal tags (like 'Harms' or 'Risks')
    useful_prefixes = ("Applications:", "Risk factors:", "Harms:", "Strategies:", "Incentives:")
    extra_labels = ["Primarily applies to the government", "Primarily applies to the private sector"]

    # Gather all columns that sound like the tags we want
    candidate_columns = [col for col in df.columns if col.startswith(useful_prefixes) or col in extra_labels]
    label_columns = []

    # Check each tag. If a tag shows up less than 3 times in the data, 
    # we throw it away because there isn't enough info for the computer to learn from it.
    for col in candidate_columns:
        positive_count = (df[col] == True).sum()
        negative_count = (df[col] == False).sum()
        if positive_count >= min_positive and negative_count >= min_positive:
            label_columns.append(col)

    return label_columns


def main():
    # Create the "models" folder if it doesn't exist yet
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading documents.csv...")
    # Open the big spreadsheet
    documents = pd.read_csv(DATA_PATH)

    print("Selecting governance tag columns...")
    # Get our filtered list of legal tags
    label_columns = get_label_columns(documents)
    print(f"Number of selected label columns: {len(label_columns)}")

    print("Creating training text...")
    # Squish the text together using the function we made above
    documents["training_text"] = make_training_text(documents)

    # 'X' is the text the computer will read
    X = documents["training_text"]
    # 'y' is the answer key (the tags the computer should guess)
    y = documents[label_columns].astype(int)

    print("Splitting train and test data...")
    # Split the data into two piles: 80% for studying (train) and 20% for testing (test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Converting text into TF-IDF features...")
    # Set up the TF-IDF tool. This is what counts the words and finds the important ones.
    vectorizer = TfidfVectorizer(
        lowercase=True, # Make all words lowercase so "Apple" and "apple" are the same
        stop_words="english", # Ignore useless words like "the", "and", "is"
        ngram_range=(1, 2), # Look at single words and two-word pairs
        max_features=10000, # Only keep the top 10,000 most common words
        min_df=1,  # Keep words even if they only show up 1 time (prevents errors on rare words)
    )

    # Teach the vectorizer what words exist in our training pile, and turn the text into math vectors
    X_train_vec = vectorizer.fit_transform(X_train)
    # Turn the test pile into math vectors using the vocabulary it just learned
    X_test_vec = vectorizer.transform(X_test)

    # Note: We train this basic guessing model (Logistic Regression) to make sure 
    # the whole system works, but our actual app uses the much smarter BART model instead.
    print("Training baseline multi-label classifier...")
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    classifier.fit(X_train_vec, y_train) # Tell the model to study the data

    print("Evaluating baseline model...")
    # Give the model a test and check its answers
    y_pred = classifier.predict(X_test_vec)
    
    # F1 score is basically a grade (like getting a B+ or an A-) on how well it guessed
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("Saving model files...")
    # Save our trained tools and tags into files (like saving your progress in a video game).
    # This allows the main app to load them up later without having to relearn everything.
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "tag_classifier.pkl"))
    joblib.dump(label_columns, os.path.join(MODEL_DIR, "label_columns.pkl"))

    print("Done. Model files saved in the models/ folder.")

# This tells the computer to start running the code from the main() block
if __name__ == "__main__":
    main()