#from typing import Union
from fastapi import FastAPI

from transformers import AutoTokenizer, BertModel
from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
import gradio as gr

config={
	"model_name":'etonkou/bert-classification',
	"n_classes": 5
}

class CustomModel(nn.Module, PyTorchModelHubMixin):
  #def __init__(self, model_name, num_classes): **kwargs
  def __init__(self, **kwargss):
    super(CustomModel, self).__init__()
    #super().__init__()
    self.pretrained_model = BertModel.from_pretrained(config['model_name'])
    self.classifier = nn.Linear(768, config['n_classes'])

  def forward(self, input_ids, attention_mask):
    output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
    output = self.classifier(output.last_hidden_state)
    return output


model_saved = CustomModel.from_pretrained("etonkou/bert-classification")
tokenizer_saved = AutoTokenizer.from_pretrained("etonkou/bert-classification")

classes = ["action","sci-fi","war","crime","romance"]


app = FastAPI()


@app.post("/predict")
def predict(text):
        with torch.no_grad():
            inputs = tokenizer_saved(text, return_tensors='pt')
            output = model_saved(inputs["input_ids"], inputs["attention_mask"])
            
            pred = torch.max(output, dim=1)

            index = pred.indices.max().item()
            if index > 4:
                index = "unknow"
            else:
                index = classes[index]

        return {"indice":pred.indices.max().item(),
            "classe":index
        }

