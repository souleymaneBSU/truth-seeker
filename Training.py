import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

#Load the dataset
import pandas as pd
data = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')

#Prepare the dataset for processing
data = data [['statement', 'tweet', 'label']]

#Split the dataset into training and test sets 
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['tweet'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42
)

#load bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_Length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)