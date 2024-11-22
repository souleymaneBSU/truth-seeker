import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Step 1: Load the dataset
# Example loading data into a pandas DataFrame
data = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')  

# Sample a subset of the data before merging
sample_frac = 0.1 # 10% sample
data = data.sample(frac=sample_frac, random_state=42)

# Step 2: Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

print("BERT Tokenizer and Model Loaded Successfully")

def get_bert_embeddings(texts):
    """Get embeddings for a list of texts using BERT."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # Average pooling of token embeddings

# Step 3: Calculate similarity between the statement and each tweet
def get_similarity(statement, tweet):
    """Calculate cosine similarity between statement and each tweet."""
    # Get embeddings for statement and tweets
    statement_embedding = get_bert_embeddings([statement])
    tweet_embeddings = get_bert_embeddings([tweet])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(statement_embedding, tweet_embeddings)
    return similarities.flatten()

# Step 4: Label statements based on voting
def classify_statement(statement, tweets, truthfulness):
    """Classify statement as true or false based on tweet similarity and voting."""
    # Calculate similarity (Assuming you have a function get_similarity already)
    threshold = 0.7
    votes={'t':0,'f':0}
    i=0
    for tweet in tweets:
        sim = get_similarity(statement, tweet)
        if sim > threshold:
            if truthfulness[i]=='Agree':
                votes['t']+=1
            elif truthfulness[i]=='Disagree':
                votes['f']+=1
        i=i+1
    #collect votes
    return "TRUE" if votes['t'] > votes['f'] else "FALSE"

# Step 5: Apply the classification to each statement
results = []

for statement in data['statement'].unique():
    # Get all tweets for the current statement
    tweets_for_statement = data[data['statement'] == statement]
    tweets = tweets_for_statement['tweet'].tolist()
    # Assuming 'majority_target' is the column that contains 3 label majority answer
    truthfulness = tweets_for_statement['3_label_majority_answer'].tolist()
    
    result = classify_statement(statement, tweets, truthfulness)
    results.append((statement, result))

# Step 6: Print the first 10 results
for i, (statement, prediction) in enumerate(results[:10]):
    print(f"Statement: {statement}\nPredicted Truthfulness: {prediction}\n")

