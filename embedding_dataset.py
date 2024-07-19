from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager 

def scrape_dynamic_website(url):
    service = Service(ChromeDriverManager().install()) 
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=ornelly/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    paragraphs = driver.find_elements(By.TAG_NAME, 'p')
    text = ' '.join([para.text for para in paragraphs])
    driver.quit()

    return text

def save_text_to_file(text,filename):
    try:
        with open(filename,'w',encoding='utf-8') as file:
            file.write(text)
            print(f"Text successfully saved to {filename}")    
    except IOError as e:
            print(f"An error occurred while saving to file: {e}")



#scrapind text from final web site
url = 'https://www.final.edu.tr/'  
scraped_text = scrape_dynamic_website(url) 
save_text_to_file(scraped_text,'scrape_text.txt')


#Embadding using BERT
import re
from transformers import BertModel, BertTokenizer
import torch 

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]','',text)
    return text

def get_document_embedding(document):
     
     document = preprocess_text(document)
     inputs = tokenizer(document, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
     with torch.no_grad():
          outputs = model(**inputs)

     # The BERT embedding is the average of the last hidden state
     last_hidden_state = outputs.last_hidden_state
     document_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
     return document_embedding 

def save_embeddings_to_file(embeddings, filename):
    torch.save(embeddings, filename)


# Example document
document = "scrape_text.txt"

embedding = get_document_embedding(document)
save_embeddings_to_file(embedding,'document_embeddings.pt')
#print(embedding)
print(f"Saved {len(embedding)} embeddings to 'document_embeddings.pt'")

# retreival-augmented question-answer

#retreival

from transformers import BertTokenizer, BertForQuestionAnswering
import torch

model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
qa_tokenizer = BertTokenizer.from_pretrained(model_name)
qa_model = BertForQuestionAnswering.from_pretrained(model_name)

def get_answer(question):
    inputs = qa_tokenizer.encode_plus(question, return_tensors='pt')
    # Tokenize input question and context and convert to PyTorch tensors
    print("Inputs:", inputs)
    
    # Check if 'input_ids' is in the dictionary
    if 'input_ids' in inputs:
        input_ids = inputs['input_ids'].tolist()[0]
    else:
        raise KeyError("key 'input_ids' not found in inputs")

    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    # Extract the start and end logits for the answer
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1
    
    # Convert token indices to answer text
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    
    return answer 

# Example usage
question = input ("enter your question")

# Get answer
answer = get_answer(question)
print(f"Answer: {answer}")

