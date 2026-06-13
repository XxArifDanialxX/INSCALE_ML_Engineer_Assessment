# INSCALE_ML_Engineer_Assessment

This is a local machine learning tool built for the ML engineer assessment. The system uses Pandas to prepare the data by merging individual legal paragraphs with their parent documents. The machine learning approach used in this system is a hybrid model consisting of a Zero-Shot Classifier called BART and a TF-IDF vectorizer. BART acts as the system's brain to understand the context of the user's input. It uses Natural Language Processing (NLP) to predict which tags apply to the queries, even if the user does not type the exact legal terminology found in the dataset. Then, the TF-IDF vectorizer turns the text into mathematical vectors to quickly scan the documents and find the exact paragraphs that match the keywords. When a user inputs a proposed AI project scenario, the system outputs a complete governance report. This output includes a clear risk verdict (High, Moderate, or Lower Risk), the specific legal governance tags triggered by the scenario, and the exact paragraphs from the legal documents that serve as supporting evidence.


How This System Works

1. Preparing the Data (`1_train_model.py`): The setup script reads the provided `documents.csv` and `segments.csv` files. It joins the specific text chunks with their parent document information so every paragraph is linked to its respective law. It then trains the TF-IDF model on this combined text and saves the tools to a local folder for fast loading.
2. Understanding the User (`2_app.py`): When the user types a scenario, the BART (Zero-Shot Classifier) model reads the input. Instead of looking for exact words, it attempts to understand the underlying meaning and predicts which governance tags apply.
3. Determining Risk Level: The system calculates a risk verdict based on the number of sensitive tags the neural network finds, instantly flagging high-risk or restricted AI implementations.
4. Searching the Documents: To retrieve the relevant text, the TF-IDF vectorizer takes the query and scans the merged dataset for exact keyword matches. It scores every paragraph to find the most mathematically similar text.
5. Giving the Response: The system outputs the risk assessment, lists the predicted governance tags, and prints out the exact text of the top legal paragraphs as proof.


Key Decisions Made

1. A Hybrid Approach (Neural Network + TF-IDF): Pure TF-IDF only finds exact words; if a user types "cops", it misses documents using the exact phrase "law enforcement". To resolve this, the `BART` deep learning model was integrated to handle tag predictions because it understands context, while TF-IDF is reserved strictly for finding text evidence.
2. High-Precision Evidence: By integrating the `segments.csv` file, the search engine does not just return a generic document summary. It points the user directly to the exact paragraph within the law that restricts or governs their scenario.


How to Run the Code

1. Create two folders in the main directory named `data` and `models`.
2. Place the provided CSV files (`documents.csv` and `segments.csv`) inside the `data` folder.
3. Open a terminal or command line.
4. Navigate (`cd`) into the project directory.
5. Run `pip install -r requirements.txt`
6. Run `python train_model.py`
7. Run `python app.py`
