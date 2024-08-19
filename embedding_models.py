from transformers import AutoTokenizer, AutoModel, DistilBertForMaskedLM
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open("trees.txt", "r", encoding="utf-8") as file:
    
    text = file.read()

# Sentences we want sentence embeddings for
#sentences = ["This is an example sentence", "Each sentence is converted","I am student at final university","i'm beatyfull","consistency in the key of success"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)

#DIstilRbert

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
#with open("trees.txt", "r", encoding="utf-8") as file:
    
    #text = file.read()

#print(text) 

#inputs = tokenizer(text, return_tensors="pt")
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, stride=50)


with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

labels = tokenizer(text, return_tensors="pt")["input_ids"]
# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)
print(outputs)



