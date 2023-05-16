#libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load BERT model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set max sequence length
MAX_SEQ_LENGTH = 128

class Model:
    def load_model(self, load_path):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f'Model loaded from <== {load_path}')
        return model

    # predict sentence label , for model 1, (0 prediction refers to bot, 1 human), 
   
  
    def predict_hate(self, model, sentence):
        tokens = tokenizer.encode_plus(
            sentence,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt')
        tokens = tokens.to(device)
        with torch.no_grad():
            outputs = model(tokens['input_ids'], token_type_ids=None, attention_mask=tokens['attention_mask'])
        logits = outputs[0]
        _, predicted = torch.max(logits, dim=1)
        return predicted.item()

    def predict_proba(self, data):
    # Load Model and Evaluate, final out put would be (0 prediction refers to bot, 1 refers to human)
        model1 = self.load_model('model_1.pt')

        predictions=[]
        for post in data:
            result1=self.predict_hate(model1, post)
            if result1==0:
                predictions.append('bot')
            else:

                predictions.append('human')
        return np.array(predictions)

# Instantiate the model
model = Model()



# Read your test data (in your data you dont need label column)
test = pd.read_csv('test_tw.csv')

## Clean the text 
test['description'] = test['description'].astype(str).str.lower()  # Convert text to lowercase
test['description'] = test['description'].str.replace(r'http\S+', 'http')  # Remove URLs while preserving "http"
test['description'] = test['description'].str.replace(r'[^\w\s#@]', '')  # Remove punctuation except hashtags
test['description'] = test['description'].str.replace(r'\n', '')  # Remove newline characters
test['description'] = test['description'].str.replace(r'\r', '')  # Remove line breaks
test['description'] = test['description'].astype(str)


predictions = model.predict_proba(test['description'][:100]) # sent your test data for prediction

# # you dont need this part since you dont have any label
accuracy = metrics.classification_report(test['label'][:100], predictions, digits=3)
print('Accuracy of model cascade: \n')
print(accuracy)

print(predictions)
