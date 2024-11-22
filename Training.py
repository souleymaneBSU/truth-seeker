import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

#Load the dataset
import pandas as pd
data = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')

#Prepare the dataset for processing
data = data [['statement', 'tweet', 'majority_target']]

# Split the data into features and labels
X = data['tweet'].tolist()  # Features (tweets)
y = data['majority_target'].tolist()  # Labels (truthfulness or other labels)

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#load bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the training texts
train_texts = X_train  # Assign the training texts to train_texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

# Check the tokenized output
print(train_encodings)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({**train_encodings, 'label': train_labels})
test_dataset = Dataset.from_dict({**test_encodings, 'label': test_labels})

# Load the BERT model for sequence classification (set num_labels based on the number of classes)
num_labels = 3  
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',              # Output directory for model checkpoints
    num_train_epochs=3,                  # Number of training epochs
    per_device_train_batch_size=8,       # Batch size for training
    per_device_eval_batch_size=16,       # Batch size for evaluation
    warmup_steps=500,                    # Number of warmup steps for the learning rate scheduler
    weight_decay=0.01,                   # Weight decay for regularization
    logging_dir='./logs',                # Directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch
)

# Set up the Trainer
trainer = Trainer(
    model=model,                         # The pre-trained model
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=test_dataset,           # Evaluation dataset
    tokenizer=tokenizer,                 # Tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print the evaluation results
print("Evaluation Results:", results)

# # Example of how to make a prediction
# new_tweet = "This is a new tweet that might agree or disagree with the statement."
# new_encodings = tokenizer([new_tweet], truncation=True, padding=True, max_length=512)

# # Use the model to predict the label
# with torch.no_grad():
#     model.eval()  # Set the model to evaluation mode
#     input_ids = torch.tensor(new_encodings['input_ids'])
#     attention_mask = torch.tensor(new_encodings['attention_mask'])
    
#     # Get predictions
#     outputs = model(input_ids, attention_mask=attention_mask)
#     logits = outputs.logits
#     predicted_label = torch.argmax(logits, dim=1)
    
#     # Map predicted label back to text (Agree, Disagree, etc.)
#     label_map = {0: 'Agree', 1: 'Disagree', 2: 'Neutral'}  
#     print("Predicted label:", label_map[predicted_label.item()])