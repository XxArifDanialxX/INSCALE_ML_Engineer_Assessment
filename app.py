import pandas as pd
import joblib
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Hide annoying red warning messages
warnings.filterwarnings("ignore")

print("Loading data and models... (This might take a minute)")

# 1. Load the pre-glued data and the saved TF-IDF search tool
merged_data = pd.read_pickle("models/merged_data.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Get a list of all the legal tags from the spreadsheet columns
legal_tags = [col for col in merged_data.columns if ":" in col]

# Turn all the paragraphs into math numbers for fast searching
document_vectors = vectorizer.transform(merged_data["search_text"])

# 2. Load the BART deep-learning module
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("\n=== System Ready! ===")

# Start the continuous loop to talk to the user
while True:
    user_input = input("\nDescribe your AI project (or type 'exit' to quit):\n> ")
    
    if user_input.lower() == 'exit':
        break
        
    print("\n[Thinking...]")
    
    # The module guesses which tags fit the user's sentence
    brain_guess = classifier(user_input, legal_tags, multi_label=True)
    
    # Keep only the tags that the module is highly confident about
    top_tags = []
    for tag, score in zip(brain_guess['labels'], brain_guess['scores']):
        if score > 0.60: 
            top_tags.append(tag)
            
    # Determine the risk level by assessing the number of dangerous tags triggered
    if len(top_tags) >= 4:
        risk_verdict = "High Risk / Likely Restricted"
    elif len(top_tags) >= 2:
        risk_verdict = "Moderate Risk / Conditionally Allowed"
    else:
        risk_verdict = "Lower Risk"
            
    # Turn the user's sentence into math numbers
    user_vector = vectorizer.transform([user_input])
    
    # Compare the user's numbers to every single paragraph to find a match
    merged_data["text_score"] = cosine_similarity(user_vector, document_vectors)[0]
    
    # Sort the spreadsheet to get the 3 highest scoring paragraphs
    best_paragraphs = merged_data.sort_values("text_score", ascending=False).head(3)
    
    # Print result
    print(f"\n=== VERDICT: {risk_verdict} ===")
    
    print("\n=== PREDICTED TAGS ===")
    if len(top_tags) == 0:
        print("No dangerous tags detected.")
    else:
        for tag in top_tags[:5]:
            print(f"- {tag}")
            
    print("\n=== EXACT LEGAL EVIDENCE ===")
    for index, row in best_paragraphs.iterrows():
        print(f"\nDocument: {row['Official name']}")
        
        # Print the exact text, but cut it off at 300 characters so it doesn't flood the screen
        exact_text = str(row['Text'])
        if len(exact_text) > 300:
            exact_text = exact_text[:300] + "..."
            
        print(f"Paragraph: \"{exact_text}\"")