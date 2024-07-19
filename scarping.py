from bs4 import BeautifulSoup
import requests

url = 'https://en.wikipedia.org/wiki/Anti-aging_product'
page = requests.get(url)
soup = BeautifulSoup(page.content,'html.parser')

scra_text = soup.get_text()


def save_text_to_file(text,filename):
    try:
        with open(filename,'w', encoding='utf-8') as file:
            file.write(text)
            print(f"Text successfully saved to {filename}")    
    except IOError as e:
            print(f"An error occurred while saving to file: {e}")

save_text_to_file(scra_text,'skin.txt')

import re
from transformers import AlbertTokenizer, AlbertModel
import torch

model_name = 'albert-base-v2'
try:
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertModel.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

def preprocess_text (text):
     text = text.lower()
     text = re.sub(r'[^a-z0-9\s]','',text)
     return text

def get_albert_embedding(document):
    try:
        inputs = tokenizer(document, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        document_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
        
        return document_embedding
    except Exception as e:
        print(f"Error during embedding generation: {e}")


def save_embeddings_to_file(embeddings, filename):
    torch.save(embeddings, filename)

document = "skin.txt"

embedding = get_albert_embedding(document)
embedding = get_albert_embedding(document)

if embedding is not None:
    print(embedding)
save_embeddings_to_file(embedding,'skin_embeddings.pt')
print(f"Saved {len(embedding)} embeddings to 'skin_embeddings.pt'")





