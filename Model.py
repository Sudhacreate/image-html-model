import torch
import torchvision
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertConfig

class CNNEncoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(CNNEncoder, self).__init__()
        self.enc_image_size = encoded_image_size
        
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        self.fine_tune()
    
    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
    
    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, max_seq_length=512):
        super(TransformerDecoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.embedding(tgt) + self.position_encoding[:, :tgt.size(1), :]
        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.fc(out)
        return out

class ImageToHTMLModel(nn.Module):
    def __init__(self, vocab_size):
        super(ImageToHTMLModel, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = TransformerDecoder(vocab_size=vocab_size)
    
    def forward(self, images, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(images)
        memory = memory.view(memory.size(0), -1, memory.size(-1))
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return out


# Define the model
        vocab_size = len(tokenizer)
        model = ImageToHTMLModel(vocab_size=vocab_size)

# Save the model architecture
        torch.save(model, 'image_to_html_model.pt')

print("Model architecture is defined and saved.")