import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

print("Step 1: Reading and merging the data...")
# Read both spreadsheets
documents = pd.read_csv("data/documents.csv")
segments = pd.read_csv("data/segments.csv")

# Glue the paragraphs (segments) to their parent documents
merged_data = segments.merge(
    documents, 
    left_on="Document ID", 
    right_on="AGORA ID", 
    how="left"
)

# Create a huge text block for each paragraph by combining the law's title, summary, and the paragraph
merged_data["search_text"] = merged_data["Official name"].fillna("") + " " + \
                             merged_data["Short summary"].fillna("") + " " + \
                             merged_data["Text"].fillna("")

print("Step 2: Training the keyword searcher (TF-IDF)...")
# Set up the search tool to ignore common filler words
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Teach the machine on the vocabulary from the text blocks
vectorizer.fit(merged_data["search_text"])

print("Step 3: Saving the model and merged data...")
os.makedirs("models", exist_ok=True)

# Save the trained search tool
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Save the glued-together data so the main app doesn't have to rebuild it every time
merged_data.to_pickle("models/merged_data.pkl")

print("Done! You can now run the app.")
