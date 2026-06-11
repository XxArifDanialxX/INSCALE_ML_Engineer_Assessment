import os
import joblib
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

warnings.filterwarnings("ignore")

# These are the folders where I keep my files
DATA_DIR = "data"
MODEL_DIR = "models"

# Exact paths to our data files and saved math models
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.csv")
SEGMENTS_PATH = os.path.join(DATA_DIR, "segments.csv")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_COLUMNS_PATH = os.path.join(MODEL_DIR, "label_columns.pkl")


def load_data():
    # This reads our big spreadsheets (CSVs)
    documents = pd.read_csv(DOCUMENTS_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    
    # This glues the text paragraphs (segments) to their main titles (documents)
    merged = segments.merge(
        documents,
        left_on="Document ID",
        right_on="AGORA ID",
        how="left",
        suffixes=("_segment", "_document"),
    )
    return documents, segments, merged


def load_model():
    # Load the saved TF-IDF tool (the exact keyword searcher)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_columns = joblib.load(LABEL_COLUMNS_PATH)
    
    print(" Loading Semantic Deep Learning Model (this takes a moment the first time)...")
    # Load the smart "brain" model that understands the meaning of sentences
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    return vectorizer, classifier, label_columns


def make_document_text(df: pd.DataFrame) -> pd.Series:
    # This takes the title, summary, and tags and squishes them together
    # into one giant block of text so it's easier to search
    text_columns = ["Official name", "Casual name", "Short summary", "Long summary", "Tags"]
    text = pd.Series("", index=df.index, dtype="object")
    
    for col in text_columns:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
            
    return text.str.strip()


def predict_tags(user_input: str, classifier, label_columns, top_k: int = 15):
    # The brain model reads the user's input and guesses the best tags.
    # It understands meaning, so it doesn't need exact word matches.
    result = classifier(user_input, label_columns, multi_label=True)
    
    # Match the tags with their confidence scores (how sure the AI is)
    tag_scores = list(zip(result['labels'], result['scores']))
    
    # Throw away any guesses that the AI is not very sure about (less than 60%)
    meaningful_tags = [(tag, score) for tag, score in tag_scores if score > 0.60]
    
    # Return the top answers
    return meaningful_tags[:top_k]


def retrieve_documents(user_input: str, predicted_tags, documents: pd.DataFrame, vectorizer, top_n: int = 5):
    # Make a safe copy of the data
    documents = documents.copy()
    
    # Get the squished text ready for searching
    documents["document_text"] = make_document_text(documents)

    # Turn the user's prompt and all the documents into math numbers (vectors)
    query_vec = vectorizer.transform([user_input])
    doc_vecs = vectorizer.transform(documents["document_text"])
    
    # Check how similar the math numbers are (Text Match Score)
    semantic_scores = cosine_similarity(query_vec, doc_vecs)[0]

    # Check if the document has the same tags that the brain model guessed (Tag Overlap Score)
    predicted_tag_names = [tag for tag, score in predicted_tags]
    tag_overlap_scores = []

    for _, row in documents.iterrows():
        overlap = sum(1 for tag in predicted_tag_names if tag in documents.columns and row.get(tag) == True)
        tag_overlap = overlap / max(len(predicted_tag_names), 1)
        tag_overlap_scores.append(tag_overlap)

    documents["semantic_score"] = semantic_scores
    documents["tag_overlap_score"] = tag_overlap_scores
    
    # Combine the scores: Give 85% importance to the exact words, and 15% to the background tags
    documents["final_score"] = (0.85 * documents["semantic_score"] + 0.15 * documents["tag_overlap_score"])
    
    # Sort them to put the highest score at the top
    return documents.sort_values("final_score", ascending=False).head(top_n)


def retrieve_segments(user_input: str, top_documents: pd.DataFrame, merged: pd.DataFrame, vectorizer, top_n: int = 3):
    # Get the ID numbers of the winning documents
    top_doc_ids = top_documents["AGORA ID"].tolist()
    
    # Find all the paragraphs that belong to those winning documents
    candidates = merged[merged["Document ID"].isin(top_doc_ids)].copy()

    # If nothing is found, just return empty
    if candidates.empty:
        return candidates

    # Throw away anything that is not related to AI
    if "Not AI-related" in candidates.columns:
        candidates = candidates[candidates["Not AI-related"] != True]

    if candidates.empty:
        return candidates

    # Squish the paragraph text and its summary together
    segment_text = candidates["Text"].fillna("").astype(str)
    if "Summary" in candidates.columns:
        segment_text = segment_text + " " + candidates["Summary"].fillna("").astype(str)

    candidates["segment_text_for_search"] = segment_text

    # Turn the prompt and paragraphs into math numbers to find the most relevant paragraph
    query_vec = vectorizer.transform([user_input])
    segment_vecs = vectorizer.transform(candidates["segment_text_for_search"])
    candidates["segment_score"] = cosine_similarity(query_vec, segment_vecs)[0]

    # Sort them so the best paragraph is at the top
    return candidates.sort_values("segment_score", ascending=False).head(top_n)


def make_assessment(predicted_tags):
    # Strip just the names from the tags
    tag_names = [tag for tag, score in predicted_tags]
    
    # A list of tags that are considered very dangerous or sensitive
    high_sensitivity_tags = {
        "Applications: Education", "Applications: Finance and investment",
        "Applications: Government: judicial and law enforcement", "Applications: Government: military and public safety",
        "Applications: Medicine, life sciences and public health", "Applications: Security",
        "Risk factors: Bias", "Risk factors: Privacy", "Risk factors: Safety", "Risk factors: Security",
        "Risk factors: Transparency", "Harms: Discrimination", "Harms: Harm to health/safety",
        "Harms: Violation of civil or human rights, including privacy",
    }

    # Count how many dangerous tags the brain model found
    matched_sensitive_tags = [tag for tag in tag_names if tag in high_sensitivity_tags]
    risk_count = len(matched_sensitive_tags)

    # Give a warning level based on how many dangerous tags were triggered
    if risk_count >= 6:
        verdict = "High-risk / likely restricted"
        confidence = "Medium"
    elif risk_count >= 3:
        verdict = "Moderate-risk / conditionally allowed"
        confidence = "Medium"
    else:
        verdict = "Lower-risk"
        confidence = "Low to Medium"

    return verdict, confidence


def main():
    # This is the starting point of the program
    print("="*60)
    print(" Loading Machine Learning Models & Data...")
    
    try:
        # Load up our data and tools
        documents, segments, merged = load_data()
        vectorizer, classifier, label_columns = load_model()
        print(" System Ready!")
    except FileNotFoundError:
        # If the files are missing, tell the user how to fix it
        print("\n [ERROR] Model or data files not found.")
        print(" Please run 'python train_model.py' first to generate the models.")
        return
    except Exception as e:
        print(f"\n [ERROR] Failed to load models: {e}")
        return

    print("="*60)
    
    # Keep asking the user for input in a continuous loop
    while True:
        print("\n" + "-"*60)
        user_input = input("Describe the AI implementation (or type 'exit' to quit):\n> ")

        # If they type exit, shut down
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Shutting down system. Goodbye!")
            break

        # If they just press enter by accident, ask again
        if not user_input.strip():
            print("Please enter a valid scenario.")
            continue

        print("\n[Processing via TF-IDF & Zero-Shot Classification...]")

        # SAFEGUARD: Check if the user used total nonsense that isn't in our dictionary
        test_vec = vectorizer.transform([user_input])
        if test_vec.nnz == 0:
            print("\n[WARNING] Your input didn't contain any keywords recognized by the AI law database.")
            print("Please try using more specific terminology.")
            continue

        # 1. Ask the models to do their jobs
        predicted_tags = predict_tags(user_input, classifier, label_columns)
        top_documents = retrieve_documents(user_input, predicted_tags, documents, vectorizer)
        top_segments = retrieve_segments(user_input, top_documents, merged, vectorizer)
        verdict, confidence = make_assessment(predicted_tags)

        # 2. Print out the results so the user can read them
        print("\n=== 1. GOVERNANCE ASSESSMENT ===")
        print(f"Verdict    : {verdict}")
        print(f"Confidence : {confidence}")

        print("\n=== 2. CATEGORIZED GOVERNANCE TAGS ===")
        # Group the tags nicely into these 5 folders
        categories = ["Applications", "Harms", "Incentives", "Risk factors", "Strategies"]
        grouped_tags = {cat: [] for cat in categories}
        
        for tag, score in predicted_tags:
            for cat in categories:
                # If the tag matches a category, clean up the name and put it in the group
                if tag.startswith(f"{cat}:"):
                    clean_tag_name = tag.replace(f"{cat}:", "").strip()
                    grouped_tags[cat].append((clean_tag_name, score))
                    break

        # Print each group out
        for cat in categories:
            print(f"\n [{cat.upper()}]")
            if not grouped_tags[cat]:
                print("   None detected")
            else:
                for clean_tag, score in grouped_tags[cat][:3]:
                    print(f"   - {clean_tag} (Confidence: {score:.2f})")

        # Print the names of the top matching laws
        print("\n=== 3. RELEVANT DOCUMENTS ===")
        for _, row in top_documents.head(3).iterrows():
            doc_name = row.get("Official name")
            doc_name = str(doc_name) if pd.notna(doc_name) else "Unknown Document"
            print(f" - {doc_name} (Score: {row.get('final_score', 0):.2f})")

        # Print the exact paragraphs that act as proof
        print("\n=== 4. SUPPORTING EVIDENCE ===")
        if top_segments.empty:
            print(" No supporting segments found.")
        else:
            for idx, row in top_segments.iterrows():
                doc_name = row.get("Official name")
                doc_name = str(doc_name) if pd.notna(doc_name) else "Unknown Document"
                
                print(f"\n[{doc_name}] - Position {row.get('Segment position', 'N/A')}")
                
                text = str(row.get("Text", ""))
                # Cut the text off if it is too long so it doesn't flood the screen
                if len(text) > 400:
                    text = text[:400] + "..."
                print(f"\"{text}\"")
                
        print("\n" + "="*60)

# This tells the computer to run the main() block when user start the script
if __name__ == "__main__":
    main()