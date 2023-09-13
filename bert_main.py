# libraires
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
#from transformers import AutoTokenizer, BertModel
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW
#import torch.optim as optim
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')


# load file 
def data_processing():    
    #path_file_name=('./wiki_movie_plots_deduped.csv')
    path_file_name ='/content/drive/MyDrive/Colab Notebooks/DeepL2/wiki_movie_plots_deduped.csv'
    df = pd.read_csv(path_file_name)
    selected_label = ['sci-fi','war','romance','crime','action']
    data = df[df['Genre'].isin(selected_label)]
    return data

def data_inputs(data):
        
    input_texts = data["Plot"].tolist()
    labels = data["Genre"].tolist()
    num_classes = len(set(labels))
    return input_texts, labels, num_classes

def labe2index_index2label(labels):
    
    label_to_index = {label: index for index, label in enumerate(set(labels))}
    index_to_label = {}
    for z in zip(label_to_index.keys(), label_to_index.values()):        
        index_to_label[z[1]] = z[0]
    return label_to_index, index_to_label

def encode_input_label(input_texts,labels,label_to_index):
    
    input_encodings = tokenizer(input_texts, truncation=True, padding=True, return_tensors="pt")
    encoded_labels = torch.tensor([label_to_index[label] for label in labels])
    return input_encodings, encoded_labels
    
def data_split(input_encodings,encoded_labels,batch_size):    
    
    data = torch.utils.data.TensorDataset(input_encodings["input_ids"], input_encodings["attention_mask"], encoded_labels)

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    train_data, test_data = random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))    

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train(model,num_epochs,lr):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    #num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            batch = tuple(t.to(device) for t in batch)
            batch_inputs, batch_attention_mask, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=batch_inputs, attention_mask = batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

def push_to_hugginface(MODEL_PATH):
    # install huggingface
    import subprocess
    #!pip install huggingface_hub
    subprocess.call(['pip', 'install', 'huggingface_hub'])
    
    #import huggingface
    from huggingface_hub import notebook_login
    
    # login to huggingface
    notebook_login()
    
    # Activate large file
    #!git lfs install
    subprocess.call(['git', 'ls', 'install'])
    
    from huggingface_hub import HfFolder
    import os
    
    # retreive token in a variable for future use
    os.environ['HF_AUTH'] = HfFolder().get_token()
    
    # use git command to set some parameters and for pushing our saved model
    #!git config --global user.email "jtonedeag@gmail.com"
    #!git config --global user.name "Koularambaye-Tonedeag"
    subprocess.call(['git', 'config', '--global', 'user.email', '"jtonedeag@gmail.com"'])
    subprocess.call(['git', 'config', '--global', 'user.name', '"Koularambaye-Tonedeag"'])
    

    # Change to the desired directory
    os.chdir(MODEL_PATH)
    
    # we clone our repo created in huggingface    
    #!git clone https://user:$HF_AUTH@huggingface.co/etonkou/bert-classification
    URL = "https://user:$HF_AUTH@huggingface.co/etonkou/bert-classification"
    subprocess.call(['git', 'clone', URL])
    
    # we run git commands in order to push our saved files
    #!git add .
    #!git commit -m "deploiement de modele bert pour la classification"
    comment="deploiement de modele bert pour la classification"
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', comment])
    subprocess.call(['git', 'push'])
    
    #!git push

if __name__ == "__main__":
    
    # parametres
    MODEL_NAME="bert-base-uncased" 
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 1
    
    data = data_processing()
    input_texts = data_inputs(data)[0]
    labels = data_inputs(data)[1]
    NUM_CLASSES = data_inputs(data)[2]
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    
    label_to_index = labe2index_index2label(labels)[0]
    index_to_label = labe2index_index2label(labels)[1]
    
    #print(label_to_index)
    input_encodings = encode_input_label(input_texts,labels,label_to_index)[0]
    encoded_labels = encode_input_label(input_texts,labels,label_to_index)[1]
    
    train_loader = data_split(input_encodings,encoded_labels,BATCH_SIZE)[0]
    test_loader = data_split(input_encodings,encoded_labels,BATCH_SIZE)[1]
    
    # train model
    train(model,NUM_EPOCHS,LEARNING_RATE)
    
    # save model
    MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/DeepL2/Bert_model'
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    # Push model to huggingface
    #push_to_hugginface(MODEL_PATH)
    
    