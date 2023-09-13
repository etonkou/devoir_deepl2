
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
 

if __name__ == "__main__":
    
    MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/DeepL2/Bert_model'    
    # Push model to huggingface
    push_to_hugginface(MODEL_PATH)
    