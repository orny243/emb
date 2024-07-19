from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def scrape_dynamic_website(url):
    # Set up the Chrome driver
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run headless Chrome (no GUI)
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
    
    driver = webdriver.Chrome(service=service, options=options)
    
    # Open the URL
    driver.get(url)
    
    # Wait for the dynamic content to load (you might need explicit waits for specific elements)
    driver.implicitly_wait(10)  # Adjust the wait time as needed
    
    # Extract content
    paragraphs = driver.find_elements(By.TAG_NAME, 'p')
    text = ' '.join([para.text for para in paragraphs])
    
    # Close the driver
    driver.quit()
    
    return text

# Example usage
url = 'https://www.final.edu.tr/'  # Replace with the website URL you want to scrape
scraped_text = scrape_dynamic_website(url)

if scraped_text is not None:
    print(scraped_text[:5])  # Print first 500 characters of the scraped text
else:
    print("Failed to scrape the website.")





#Embadding using BERT

from transformers import BertModel, BertTokenizer
import torch 

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def get_document_embedding(document):
    # Tokenize input text and convert to PyTorch tensors
    inputs = tokenizer(document, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The BERT embedding is the average of the last hidden state
    last_hidden_state = outputs.last_hidden_state
    document_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
    
    return document_embedding

# Example document
document = "This is an example of an academic document about natural language processing."

# Get embedding
embedding = get_document_embedding(document)
print(embedding)


    
    


