
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

def predict(text):
    with torch.no_grad():
        inputs = tokenizer_saved(text, return_tensors='pt')
        output = model_saved(inputs["input_ids"], inputs["attention_mask"])
        #print(f"the output {output}")
        pred = torch.max(output, dim=1)
        #print(f"the pred {pred.indices.max().item()}")
        #print(f"indice : {pred.indices}")
        index = pred.indices.max().item()
        if index > 4:
            index = "unknow"
        else:
            index = classes[index]

    return {"indice":pred.indices.max().item(),
           "classe":index
    }

# p = predict('it\'s start with people killed by a robot')
# print(p)
demo = gr.Interface(fn=predict, inputs="text", outputs="json")
demo.launch()


# def greet(name):
#     return "Hello " + name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

