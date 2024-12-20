import torch
from abc import ABC
from transformers import AutoTokenizer, MT5EncoderModel, AutoModel

import re
from NitUtils import TokenizerOutputs, EmbeddingsOutputs
from torch import nn

class NitEncoder(ABC):

    def __init__(self, model, tokenizer, max_tokens = 100 , padding='max_length'):
        self.model = model
        self.tokenizer : AutoTokenizer = tokenizer
        self.embedding_size = self.model.config.hidden_size
        self.embedding_layer = self.model.get_input_embeddings()
        self.max_tokens = max_tokens
        self.padding = padding
        self.template = None
        self.device_me = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device_me)
    
    def useDataParellel(self, gpu_ids):
        self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        self.embedding_layer = nn.DataParallel(self.embedding_layer, device_ids=gpu_ids)
        
    def setPadding(self,padding):
        self.padding = padding

    def setTemplate(self,template):
        self.template = template

    def setMaxTokens(self,max_tokens):  
        self.max_tokens = max_tokens

    def to(self, device):
        self.model.to(device)
        self.embedding_layer.to(device)
        self.device_me = device

    def getEmbedding_dim(self):
        return self.embedding_size
    
    def getEmbedding_shape(self):
        return (self.max_tokens,self.embedding_size)
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def applyTemplate(self,texts):
        if self.template is not None:
            return [self.template.format(text=text) for text in texts]
        return texts
    
    def applyFormating(self,texts):
        return texts
    
    def get_inputIds(self, texts):
        texts = self.applyFormating(texts)
        texts = self.applyTemplate(texts)
        encoding = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=self.padding,
            truncation=True,
            max_length=self.max_tokens
        )
        output = TokenizerOutputs(encoding["input_ids"].to(self.device_me), encoding["attention_mask"].to(self.device_me))
        return output
    
    def get_embeddings(self, inputs : TokenizerOutputs, **kwargs):
        return EmbeddingsOutputs(self.embedding_layer(inputs.input_ids), inputs.attention_mask)

    def get_embeddings_from_text(self, texts, **kwargs):
        return self.get_embeddings(self.get_inputIds(texts), **kwargs)
    
class NitMT5encoder(NitEncoder):
    def __init__(self,cache_dir, max_tokens = 100, padding='max_length'):
        model = MT5EncoderModel.from_pretrained("google/mt5-large", cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-large", cache_dir = cache_dir, legacy=True, padding_side='left')
        super().__init__(model, tokenizer, max_tokens, padding)

    def get_embeddings(self, inputs : TokenizerOutputs, **kwargs):
        hidden_layer = kwargs.get("hiddenLayer", True)
        if hidden_layer:
            attention_mask = inputs.attention_mask
            input_ids = inputs.input_ids
            embeddings =  self.model(input_ids =input_ids,
                                    attention_mask = attention_mask, 
                                    return_dict=True, 
                                    output_hidden_states=True).last_hidden_state
            return EmbeddingsOutputs(embeddings, attention_mask)
        else:
            return super().get_embeddings(inputs, **kwargs)

class NitRobertaencoder(NitEncoder):
    def __init__(self,cache_dir, max_tokens = 100, padding='max_length'):
        model = AutoModel.from_pretrained("FacebookAI/roberta-base", cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", cache_dir = cache_dir, padding_side='right')
        super().__init__(model, tokenizer, max_tokens, padding)

    def get_embeddings(self, inputs : TokenizerOutputs, **kwargs):
        hidden_layer = kwargs.get("hiddenLayer", True)
        cls_token = kwargs.get("cls_token", False)
        pooler_output = kwargs.get("pooler_output", False)
        if hidden_layer or cls_token or pooler_output:
            attention_mask = inputs.attention_mask
            input_ids = inputs.input_ids
            embeddings =  self.model(input_ids =input_ids,
                                    attention_mask = attention_mask, 
                                     return_dict=True, 
                                     output_hidden_states=True)
            if pooler_output:
                embeddings = embeddings.pooler_output
            
            elif cls_token:
                embeddings = embeddings.last_hidden_state[:,0,:]
            else:
                embeddings = embeddings.last_hidden_state
            return EmbeddingsOutputs(embeddings, attention_mask)
        else:
            return super().get_embeddings(inputs, **kwargs)
        
    def applyFormating(self, texts):
        return [self.split_numbers_in_text(text) for text in texts]
    
    def split_numbers_in_text(self,text):
        return re.sub(r'\d+', lambda x: f" {' '.join(x.group())}", text)