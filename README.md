# INSCALE_ML_Engineer_Assessment

This is the machine learning tool i built for the given ML engineer assessment. So this system use Pandas to prepare the data by gluing all the paragraphs together with their tags. the machine learning approach is used in this system is a hybrid approach consisting of a Zero-Shot Classifier called BART and the TF-IDF vectorizer. the BART will act as the system's brain to understand the context of the input from the user where it uses NLP to predict which tags apply to the queries even if the user doesn't type the exact word like in the dataset. then, the TF-IDF vectorizer will turn the text into mathematical vectors to quickly scan the documents and find the exact paragraphs that match the user's keywords. finally, the system will combine the BART part to predict the tags for the user queries with the TF-IDF that search the words in the dataset. When a user inputs a proposed AI project scenario, the system outputs a complete governance report. This output includes a clear risk verdict (High, Moderate, or Lower Risk), the specific legal governance tags triggered by the scenario, and the most relevant legal documents alongside the exact paragraphs that serve as supporting evidence.

How This System Works:
1. Loading the Data: The script (cli.py) reads the provided `documents.csv` and `segments.csv` files. It joins the specific text chunks with their parent document information so every paragraph knows what law it belongs to.
2. Understanding the User: When the user type a scenario, the BART (Zero-Shot Classifier) will reads the input. Instead of looking for exact words, it try to understands the meaning and predicts which governance tags apply to the input
3. Searching the Documents: To actually find the right text in the documents, I use a TF-IDF vectorizer. It takes the query and scans the dataset for exact keyword matches. The system then combines this text-match score (85% weight) with the tag-match score (15% weight) to rank the top documents.
4. Giving the Response: Based on how many sensitive tags the neural network found, the system outputs a simple risk assessment, which governance tags are related to it and prints out the top 3 paragraphs of legal text as proof.

Key Decisions I made:
1. A Hybrid Approach (Neural Network + TF-IDF): I initially tried using TF-IDF for everything. The problem is that TF-IDF only finds exact words. If a user types "cops", it completely misses documents that use the word "law enforcement". To solve this, I added the `BART` deep learning model to handle the tag predictions because it actually understands context.
2. Weighted Scoring: I weighted the final retrieval score heavily toward the actual text match (0.85) rather than the background tags (0.15). Since user prompts are short, letting the background tags weigh too much can cause the system to drift away from what the user actually asked.

Limitations:
1. Limitations of number of words: Because the document text search still relies on TF-IDF, it can sometimes miss relevant paragraphs if the user doesn't type any of the exact keywords used in the legal text. Also, the dataset needs to have a lot more laws so there will be more information.

How to run the code:
1. create 2 files and name them 'data' and 'models', put the csv files in 'data' folder.
2. open terminal/command line
3. cd to the file directories
4. run "pip install -r requirements.txt"
5. run "python train_model.py"
6. run "python app.py"
