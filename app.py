import os
import joblib
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"

DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.csv")
SEGMENTS_PATH = os.path.join(DATA_DIR, "segments.csv")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_COLUMNS_PATH = os.path.join(MODEL_DIR, "label_columns.pkl")


def load_data():
    documents = pd.read_csv(DOCUMENTS_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    merged = segments.merge(
        documents,
        left_on="Document ID",
        right_on="AGORA ID",
        how="left",
        suffixes=("_segment", "_document"),
    )
    return documents, segments, merged


def load_model():
    # Load the TF-IDF Vectorizer for text retrieval
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_columns = joblib.load(LABEL_COLUMNS_PATH)
    
    print(" Loading Semantic Deep Learning Model (this takes a moment the first time)...")
    # Load the Zero-Shot Classifier for tag prediction
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    return vectorizer, classifier, label_columns


def make_document_text(df: pd.DataFrame) -> pd.Series:
    text_columns = ["Official name", "Casual name", "Short summary", "Long summary", "Tags"]
    text = pd.Series("", index=df.index, dtype="object")
    for col in text_columns:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
    return text.str.strip()


def predict_tags(user_input: str, classifier, label_columns, top_k: int = 15):
    # Deep learning model understands semantic meaning rather than exact keywords
    result = classifier(user_input, label_columns, multi_label=True)
    tag_scores = list(zip(result['labels'], result['scores']))
    
    # Filter out low-confidence guesses
    meaningful_tags = [(tag, score) for tag, score in tag_scores if score > 0.60]
    return meaningful_tags[:top_k]


def retrieve_documents(user_input: str, predicted_tags, documents: pd.DataFrame, vectorizer, top_n: int = 5):
    documents = documents.copy()
    documents["document_text"] = make_document_text(documents)

    query_vec = vectorizer.transform([user_input])
    doc_vecs = vectorizer.transform(documents["document_text"])
    semantic_scores = cosine_similarity(query_vec, doc_vecs)[0]

    predicted_tag_names = [tag for tag, score in predicted_tags]
    tag_overlap_scores = []

    for _, row in documents.iterrows():
        overlap = sum(1 for tag in predicted_tag_names if tag in documents.columns and row.get(tag) == True)
        tag_overlap = overlap / max(len(predicted_tag_names), 1)
        tag_overlap_scores.append(tag_overlap)

    documents["semantic_score"] = semantic_scores
    documents["tag_overlap_score"] = tag_overlap_scores
    
    # FIXED Weights: Heavily favor the actual text match (0.85) over background tags (0.15)
    documents["final_score"] = (0.85 * documents["semantic_score"] + 0.15 * documents["tag_overlap_score"])
    
    return documents.sort_values("final_score", ascending=False).head(top_n)


def retrieve_segments(user_input: str, top_documents: pd.DataFrame, merged: pd.DataFrame, vectorizer, top_n: int = 3):
    top_doc_ids = top_documents["AGORA ID"].tolist()
    candidates = merged[merged["Document ID"].isin(top_doc_ids)].copy()

    if candidates.empty:
        return candidates

    if "Not AI-related" in candidates.columns:
        candidates = candidates[candidates["Not AI-related"] != True]

    if candidates.empty:
        return candidates

    segment_text = candidates["Text"].fillna("").astype(str)
    if "Summary" in candidates.columns:
        segment_text = segment_text + " " + candidates["Summary"].fillna("").astype(str)

    candidates["segment_text_for_search"] = segment_text

    query_vec = vectorizer.transform([user_input])
    segment_vecs = vectorizer.transform(candidates["segment_text_for_search"])
    candidates["segment_score"] = cosine_similarity(query_vec, segment_vecs)[0]

    return candidates.sort_values("segment_score", ascending=False).head(top_n)


def make_assessment(predicted_tags):
    tag_names = [tag for tag, score in predicted_tags]
    high_sensitivity_tags = {
        "Applications: Education", "Applications: Finance and investment",
        "Applications: Government: judicial and law enforcement", "Applications: Government: military and public safety",
        "Applications: Medicine, life sciences and public health", "Applications: Security",
        "Risk factors: Bias", "Risk factors: Privacy", "Risk factors: Safety", "Risk factors: Security",
        "Risk factors: Transparency", "Harms: Discrimination", "Harms: Harm to health/safety",
        "Harms: Violation of civil or human rights, including privacy",
    }

    matched_sensitive_tags = [tag for tag in tag_names if tag in high_sensitivity_tags]
    risk_count = len(matched_sensitive_tags)

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
    print("="*60)
    print(" Loading Machine Learning Models & Data...")
    
    try:
        documents, segments, merged = load_data()
        vectorizer, classifier, label_columns = load_model()
        print(" System Ready!")
    except FileNotFoundError:
        print("\n [ERROR] Model or data files not found.")
        print(" Please run 'python train_model.py' first to generate the models.")
        return
    except Exception as e:
        print(f"\n [ERROR] Failed to load models: {e}")
        return

    print("="*60)
    
    while True:
        print("\n" + "-"*60)
        user_input = input("Describe the AI implementation (or type 'exit' to quit):\n> ")

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Shutting down system. Goodbye!")
            break

        if not user_input.strip():
            print("Please enter a valid scenario.")
            continue

        print("\n[Processing via TF-IDF & Zero-Shot Classification...]")

        # SAFEGURARD: Check if the vectorizer recognized ANY words in the user's input
        test_vec = vectorizer.transform([user_input])
        if test_vec.nnz == 0:
            print("\n[WARNING] Your input didn't contain any keywords recognized by the AI law database.")
            print("Please try using more specific terminology.")
            continue

        # 1. Run Predictions
        predicted_tags = predict_tags(user_input, classifier, label_columns)
        top_documents = retrieve_documents(user_input, predicted_tags, documents, vectorizer)
        top_segments = retrieve_segments(user_input, top_documents, merged, vectorizer)
        verdict, confidence = make_assessment(predicted_tags)

        # 2. Display Output
        print("\n=== 1. GOVERNANCE ASSESSMENT ===")
        print(f"Verdict    : {verdict}")
        print(f"Confidence : {confidence}")

        print("\n=== 2. CATEGORIZED GOVERNANCE TAGS ===")
        categories = ["Applications", "Harms", "Incentives", "Risk factors", "Strategies"]
        grouped_tags = {cat: [] for cat in categories}
        
        for tag, score in predicted_tags:
            for cat in categories:
                if tag.startswith(f"{cat}:"):
                    clean_tag_name = tag.replace(f"{cat}:", "").strip()
                    grouped_tags[cat].append((clean_tag_name, score))
                    break

        for cat in categories:
            print(f"\n [{cat.upper()}]")
            if not grouped_tags[cat]:
                print("   None detected")
            else:
                for clean_tag, score in grouped_tags[cat][:3]:
                    print(f"   - {clean_tag} (Confidence: {score:.2f})")

        print("\n=== 3. RELEVANT DOCUMENTS ===")
        for _, row in top_documents.head(3).iterrows():
            doc_name = row.get("Official name")
            doc_name = str(doc_name) if pd.notna(doc_name) else "Unknown Document"
            print(f" - {doc_name} (Score: {row.get('final_score', 0):.2f})")

        print("\n=== 4. SUPPORTING EVIDENCE ===")
        if top_segments.empty:
            print(" No supporting segments found.")
        else:
            for idx, row in top_segments.iterrows():
                doc_name = row.get("Official name")
                doc_name = str(doc_name) if pd.notna(doc_name) else "Unknown Document"
                
                print(f"\n[{doc_name}] - Position {row.get('Segment position', 'N/A')}")
                
                text = str(row.get("Text", ""))
                if len(text) > 400:
                    text = text[:400] + "..."
                print(f"\"{text}\"")
                
        print("\n" + "="*60)

if __name__ == "__main__":
    main()