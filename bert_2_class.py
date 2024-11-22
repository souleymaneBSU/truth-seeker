import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Truth_Seeker_Model_Dataset.csv")  # Replace with actual dataset path

# Ensure no statement_id overlap in train/test split
def split_dataset(data, test_size=0.2):
    unique_statements = data["statement"].unique()
    train_statements, test_statements = train_test_split(unique_statements, test_size=test_size, random_state=42)
    
    train_data = data[data["statement"].isin(train_statements)]
    test_data = data[data["statement"].isin(test_statements)]
    return train_data, test_data

train_data, test_data = split_dataset(data)

# Map string labels to integers
def map_labels(data, label_column):
    unique_labels = data[label_column].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    data[label_column] = data[label_column].map(label_mapping)
    return data, label_mapping

# Map 3-label data 
train_data, train_3_label_mapping = map_labels(train_data, "3_label_majority_answer")
test_data, _ = map_labels(test_data, "3_label_majority_answer")
print("3-label Mapping:", train_3_label_mapping)


# Dataset Class for PyTorch
class TruthSeekerDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            tweet,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Prepare data for 5-label classification
def prepare_datasets(train_data, test_data, label_type, tokenizer):
    train_dataset = TruthSeekerDataset(
        tweets=train_data["tweet"].tolist(),
        labels=train_data[label_type].tolist(),
        tokenizer=tokenizer
    )
    test_dataset = TruthSeekerDataset(
        tweets=test_data["tweet"].tolist(),
        labels=test_data[label_type].tolist(),
        tokenizer=tokenizer
    )
    return train_dataset, test_dataset

# Compute metrics function for accuracy
def compute_metrics(pred):
    predictions, labels = pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc
    }

# Model training function
def train_bert(train_dataset, test_dataset, num_labels):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir="./bert_model",  # Temporary directory
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="no",  # Avoid saving any checkpoints
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics  # Pass compute_metrics function
    )
    trainer.train()
    
    # Get the final evaluation metrics
    final_metrics = trainer.evaluate()
    final_accuracy = final_metrics["eval_accuracy"]
    return final_accuracy

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#3-label Classification (Commented out)
print("Training 3-label classification...")
train_3_label, test_3_label = prepare_datasets(train_data, test_data, "3_label_majority_answer", tokenizer)
final_accuracy_3_label = train_bert(train_3_label, test_3_label, num_labels=3)

# Final Output for 3-label
print(f"Final accuracy for 3-label classification: {final_accuracy_3_label}")
