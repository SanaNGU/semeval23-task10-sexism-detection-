import pandas as pd
import torch
import wandb
import numpy as np
from torch import nn
from transformers import AdamW,AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding,AutoConfig
from torch.utils.data import DataLoader
from datasets import load_dataset
unlabeled_data=pd.read_csv('/home/sanala/Juputer try/detection of sexism/starting_ki/gab_1M_unlabelled.csv')

from scipy.special import softmax as sx

class Dataset(torch.utils.data.Dataset):    
    def __init__(self, encodings, labels=None):          
        self.encodings = encodings        
        self.labels = labels
     
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    
checkpoint='vinai/bertweet-large'#vinai/bertweet-base
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-large-ot')
test_trainer = Trainer(model)



X_test = list(unlabeled_data["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)



predictions = test_trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = pd.Series(preds).map(model.config.id2label)
smax = np.array([list(sx(item)) for item in predictions[0]])



#test_trainer.predict(test_dataset)

weak_labeled_data=pd.read_csv('/home/sanala/Juputer try/final_EDOS/gab_1M_unlabelled.csv')
weak_labeled_data['label_sexist']=labels
weak_labeled_data['softmax']=smax
weak_labeled_data.to_csv('/home/sanala/Juputer try/final_EDOS/gab_1M_weak_labels.csv', index=False)
weak_labeled_data.drop(weak_labeled_data[weak_labeled_data['softmax'] <= 0.7].index, inplace = True)
weak_labeled_data.drop(weak_labeled_data[weak_labeled_data['label_sexist'] == 'not sexist'].index, inplace = True)
weak_labeled_data.to_csv('/home/sanala/Juputer try/final_EDOS/gab_1M_weak_labels_sexiest.csv', index=False)

