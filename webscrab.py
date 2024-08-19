from transformers import AutoTokenizer, DistilBertForMaskedLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

# Read the content of the text file
with open("trees.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the input text in chunks
tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

# Process in chunks of maximum length
chunk_size = 512
for i in range(0, len(tokens), chunk_size):
    chunk = tokens[i:i + chunk_size]
    chunk_inputs = {"input_ids": chunk.unsqueeze(0)}

    # Perform inference without computing gradients
    with torch.no_grad():
        logits = model(**chunk_inputs).logits

    # Process logits as needed (e.g., predictions)
    # Note: Adjust processing logic according to your task
mask_token_index = (chunk_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

labels = tokenizer(text, return_tensors="pt")["input_ids"]
# mask labels of non-[MASK] tokens
labels = torch.where(chunk_inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**chunk_inputs, labels=labels)
print(outputs)

