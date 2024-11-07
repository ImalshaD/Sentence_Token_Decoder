import torch
from abc import ABC
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..NitUtils import TokenizerOutputs, EmbeddingsOutputs

class NitEncoder(ABC):

    def __init__(self, model, tokenizer, max_tokens = 100 , padding='max_length'):
        self.model = model
        self.tokenizer : AutoTokenizer = tokenizer
        self.embedding_size = self.model.config.hidden_size
        self.embedding_layer = self.model.get_input_embeddings()
        self.max_tokens = max_tokens
        self.padding = padding

    def setPadding(self,padding):
        self.padding = padding

    def setMaxTokens(self,max_tokens):  
        self.max_tokens = max_tokens

    def to(self, device):
        self.model.to(device)
        self.embedding_layer.to(device)
        self.device_me = device
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_inputIds(self, texts):
        encoding = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=self.padding,
            truncation=True,
            max_length=self.max_tokens
        )
        output = TokenizerOutputs(encoding["input_ids"].to(self.device_me), encoding["attention_mask"].to(self.device_me))
        return output
    
    def get_embeddings(self, inputs : TokenizerOutputs):
        return EmbeddingsOutputs(self.embeddings(inputs.input_ids), inputs.attention_mask)

    def original_pipeline(self, texts):
        return self.get_embeddings(self.get_inputIds(texts))
    
class NitMT5encoder(NitEncoder):
    def __init__(self,cache_dir):
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large", cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-large", cache_dir = cache_dir, legacy=False, padding_side='left')
        super().__init__(model, tokenizer)