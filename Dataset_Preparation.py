import os
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the WebSight dataset from Hugging Face
dataset = load_dataset('Dataset card')

# Split the dataset into training, validation, and test sets
train_val_data, test_data = train_test_split(dataset['train'], test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)

# Normalize and augment the images if necessary
def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

# Tokenize the HTML code
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class WebSightDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = preprocess_image(item['image'])
        html = item['html']
        html_tokens = self.tokenizer(html, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return torch.tensor(image, dtype=torch.float32), html_tokens.input_ids.squeeze(), html_tokens.attention_mask.squeeze()

train_dataset = WebSightDataset(train_data, tokenizer)
val_dataset = WebSightDataset(val_data, tokenizer)
test_dataset = WebSightDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Save the loaders for later use
torch.save(train_loader, 'train_loader.pt')
torch.save(val_loader, 'val_loader.pt')
torch.save(test_loader, 'test_loader.pt')

print("Dataset preparation is complete.")