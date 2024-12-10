#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


from torch.utils.data import DataLoader, TensorDataset


# In[4]:


from datasets import load_dataset


# In[5]:


from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from pytorch_lightning.callbacks import ModelCheckpoint


# In[6]:


from tqdm import tqdm


# In[7]:


from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)


# In[8]:


from einops import rearrange


# In[9]:


import re


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


import os


# # Dataset loading and Preprocessing

# In[14]:
torch.set_float32_matmul_precision('high')

def extract_num_output(text):
    match = re.search(r'(?<=The answer is:\s).*$', text)
    if match:
        return match.group(0)
    return None


# In[15]:


cache_dir = "data_cache"
model_dir = "model_cache"
ds = load_dataset("meta-math/MetaMathQA", cache_dir=cache_dir)


# In[16]:


df = pd.DataFrame(ds['train']) 


# In[17]:


df['Numerical_output']= df['response'].apply(extract_num_output)


# # Main Classes

# In[18]:


class TokenizerOutputs:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


# In[19]:


class EmbeddingsOutputs:
    def __init__(self, input_embeds, attention_mask):
        self.input_embeds = input_embeds
        self.attention_mask = attention_mask


# In[20]:


class AlignmentModel(ABC, nn.Module):
    input_shape : tuple
    output_shape : tuple

    def __init__(self,input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, x):
        pass

    def test(self):
        sample_input  = np.zeros(self.input_shape)
        output = self.forward(sample_input)
        assert output.shape == self.output_shape


# In[21]:


class NITConfig(PretrainedConfig):
    modelName = "NITModel"

    def __init__(self,
                 llm_name="Qwen/Qwen-1.5B",
                 llm_dim = 1536,
                 freez_encoder= True,
                 freez_llm = True,
                 alignment_class= None,
                 max_tokens = 200,
                 model_cache_dir = "model_cache",
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm_name
        self.freez_encoder = freez_encoder
        self.freez_llm = freez_llm
        self.alignment_class = alignment_class
        self.max_tokens = max_tokens
        self.model_cache =model_cache_dir
        self.llm_dim = llm_dim


# In[22]:


class NITModel(PreTrainedModel):
    tokenizer : AutoTokenizer
    llm : AutoModelForCausalLM
    alingment_model : AlignmentModel
    embeddings: nn.Embedding

    def __init__(self, config : NITConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm,cache_dir=config.model_cache, padding_side='left')
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm,cache_dir=config.model_cache)
        self.input_shape = (config.max_tokens,config.llm_dim)
        self.output_shape = (config.max_tokens,config.llm_dim)
        self.llm_dim = config.llm_dim
        self.maxTokens = config.max_tokens
        self.alingment_model = config.alignment_class(self.input_shape,self.output_shape)
        self.device_me = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.llm.to(self.device_me)
        # self.alingment_model.to(self.device_me)
        self.embeddings = self.llm.get_input_embeddings()

        if config.freez_llm:
            print("LLM Forzen")
            self.freeze_lm()
        else:
            print("LLM not Frozen")
            
        self.log_params(self.alingment_model)

    def count_parameters(self,model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params

    def log_params(self,model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print("Total trainable params in alignment Module", trainable_params)
        print("Total Non-trainable params in alignment Module", non_trainable_params)

    def freeze_lm(self):
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def unfreeze_lm(self):
        for param in self.llm.parameters():
            param.requires_grad = True
    
    def get_inputIds(self, texts):
        encoding = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.maxTokens
        )
        output = TokenizerOutputs(encoding["input_ids"].to(self.device_me), encoding["attention_mask"].to(self.device_me))
        return output
    
    def get_embeddings(self, inputs : TokenizerOutputs):
        return EmbeddingsOutputs(self.embeddings(inputs.input_ids), inputs.attention_mask)
    
    def get_embeddings_altered(self, inputs : TokenizerOutputs):
        return EmbeddingsOutputs(self.alingment_model(self.get_embeddings(inputs).input_embeds), inputs.attention_mask)
    
    def original_pipeline(self, texts):
        return self.get_embeddings(self.get_inputIds(texts))
    
    def altered_pipeline(self, texts):
        return self.get_embeddings_altered(self.get_inputIds(texts))

    def get_outputs(self, embeddings : EmbeddingsOutputs):
        return self.llm(
            attention_mask=embeddings.attention_mask,
            inputs_embeds=embeddings.input_embeds,
            return_dict=True
        )
    
    def pipeline_outputs(self, texts, altered = False, grad = False):
        if altered:
            return self.get_outputs(self.altered_pipeline(texts))
        return self.get_outputs(self.original_pipeline(texts))
    
    def forward(self , texts):
        
        # TODO: Modify accordingly
        batch_size = len(texts)

        original_output = self.pipeline_outputs(texts)
        altered_outputs = self.pipeline_outputs(texts, altered=True)

        original_logits = original_output.logits
        altered_logits = altered_outputs.logits

        original_prob = F.softmax(original_logits, dim=1)
        original_labels = torch.argmax(original_prob, dim=1)

        loss = F.cross_entropy(altered_logits,original_labels)
        
        # loss = rearrange(loss, '(b s) -> b s', b=batch_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=altered_logits,
            hidden_states=altered_outputs.hidden_states,
            attentions=altered_outputs.attentions,
        )


# # Params

# In[23]:


batch_size = 16
max_tokens = 100
embedding_lenght = 1536


# # Dataloaders

# In[24]:


split=1
split_size = 395000


# In[25]:


X_train_query = df["query"][split_size*(split-1):split_size*(split)]


# In[26]:


data_array = X_train_query.to_numpy()

# Split the data into train and test sets
X_train, X_test = train_test_split(data_array, test_size=0.2, random_state=42)

# Create DataLoaders
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=11)
test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=11)


# # Alignment Model Implementations

# In[27]:


class SimpleAlignmentModel(AlignmentModel):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        # Calculate flattened dimensions for linear transformation
        print("Testing Purpose only Don't use this model.")
        input_dim = input_shape[0] * input_shape[1]
        output_dim = output_shape[0] * output_shape[1]
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim, 1000)
        self.linear2 = nn.Linear(1000, output_dim)

    def forward(self, x):
        # Flatten input to match linear layer's expected shape
        x = self.flatten(x)
        # Pass through linear layer
        x = self.linear1(x)
        x = self.linear2(x)
        
        # Reshape output to match output_shape
        x = x.view(-1,self.output_shape[0],self.output_shape[1])  
        return x


# In[28]:


class LinearAlignementModel(AlignmentModel):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        input_dim = input_shape[0] * input_shape[1]
        output_dim = output_shape[0] * output_shape[1]
        self.flatten = nn.Flatten()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024*2),  # Input size: 200*1536, Output size: 1024
            nn.LeakyReLU(0.1),
            nn.Linear(1024*2,1024),
            # nn.LeakyReLU(0.001),
            # nn.Linear(1024*2, 1024)             # Bottleneck size: 64
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(1024, 1024*2),             # Input size: 64, Output size: 256
            # nn.LeakyReLU(0.001),
            nn.Linear(1024,1024*2),
            nn.LeakyReLU(0.1),
            nn.Linear(1024*2, output_dim),    # Output size: 200*1536
            nn.Tanh()                  #
        )
    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1,self.output_shape[0],self.output_shape[1])   # Reshape to original image size
    
class Conv1dAutoencoder(AlignmentModel):
    def __init__(self, input_shape, output_shape):
        super().__init__((100,1536), (100,1536))
        
        # Encoder
        self.encoder = nn.Sequential(
            # First layer to reduce to (100, 800)
            nn.Conv1d(in_channels=1536, out_channels=800, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(in_channels=800, out_channels=400, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            # Second layer to reduce to (100, 100)
            nn.Conv1d(in_channels=400, out_channels=200, kernel_size=1),
            nn.LeakyReLU(0.1),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            
            # First layer to expand back to (100, 800)
            nn.ConvTranspose1d(in_channels=200, out_channels=400, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose1d(in_channels=400, out_channels=800, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            # Second layer to expand back to (100, 1536)
            nn.ConvTranspose1d(in_channels=800, out_channels=1536, kernel_size=1),
            nn.Tanh()  # Use Tanh to output values in the range [-1, 1]
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100*200, 2048)
        self.fc2 = nn.Linear(2048, 100*200)
        
    def forward(self, x):
        # Transpose to (batch_size, in_channels, sequence_length)
        x = x.transpose(1, 2)  # Shape becomes (batch_size, 1536, 100)
        
        # Encode
        encoded = self.encoder(x)  # Shape becomes (batch_size, 100, 100)

        encoded_flatten = self.flatten(encoded)

        bottle_neck = self.fc1(encoded_flatten)

        decoded_fc = self.fc2(bottle_neck)

        encoded = decoded_fc.view(-1, 200, 100)
        
        # Decode
        decoded = self.decoder(encoded)  # Shape becomes (batch_size, 1536, 100)
        
        # Transpose back to (batch_size, sequence_length, feature_dimension)
        decoded = decoded.transpose(1, 2)  # Final shape (batch_size, 100, 1536)
        
        return decoded

# # Utility Functions of save/load Model/Checkpoints

# In[29]:


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


# In[30]:


def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filename}, starting at epoch {start_epoch} with loss {loss}")
        return start_epoch, loss
    else:
        print(f"No checkpoint found at {filename}")
        return 0, None


# In[31]:


# Function to save the model
def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


# In[32]:


# Function to load the model
def load_model(model, filename='model.pth'):
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filename}")


# # Training NIT

# In[33]:


config = NITConfig(
    llm_name="Qwen/Qwen2.5-Math-1.5B",
    llm_dim=embedding_lenght,
    freeze_encoder=True,
    freez_llm=True,
    alignment_class=Conv1dAutoencoder,
    max_tokens=max_tokens,
    model_cache_dir= model_dir
)


# In[34]:


nit_model = NITModel(config)


# In[35]:


# Training function with GPU support
def train_model(model: NITModel, train_loader, val_loader, optimizer,start_epoch=0 ,num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(start_epoch,num_epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        
        # Training phase
        for inputs in tqdm(train_loader, desc = f'epoch_{epoch+1}/{num_epochs}: loss {train_loss}'):
            
            batch_size = len(inputs)

            loss = model.forward(inputs).loss
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size  # Accumulate training loss
            
        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs in val_loader:
                batch_size = len(inputs)
                loss = model.forward(inputs).loss
                val_loss += loss.item() * batch_size  # Accumulate validation loss
        
        # Calculate average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        try:
            save_checkpoint(nit_model,optimizer,epoch,train_loss)
        except:
            print('Checkpoint saving Failed.')

        try:
            save_model(model)
        except:
            print("Saving Model failed")
        # Print losses
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")


# In[36]:


optimizer = torch.optim.Adam(nit_model.parameters(), lr=1e-3)


# In[37]:


start_epoch,_  = load_checkpoint(nit_model,optimizer)


# In[38]:


# train_model(nit_model, train_loader, test_loader, optimizer,start_epoch=start_epoch ,num_epochs=1)


# # Trainig With Pytorch-lightining

# In[39]:


import pytorch_lightning as pl


# In[40]:


class NITModelLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super(NITModelLightning, self).__init__()
        self.model = model
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss = self.model(batch).loss
        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(batch).loss
        self.log("val_loss", loss.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


# In[41]:

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",            # metric to monitor
    dirpath="checkpoints/",         # directory to save the checkpoints
    filename="best-checkpoint",     # filename for the saved model
    save_top_k=1,                   # save only the top model based on the monitored metric
    mode="min"                      # minimize the monitored metric (val_loss)
)

trainer = pl.Trainer(
    max_epochs=4,
    accelerator='gpu',  # This will automatically choose 'gpu' if available, else 'cpu'
    devices= 4 if torch.cuda.is_available() else None,  # Use 1 GPU if available
    log_every_n_steps=100,  # Log progress every 10 steps
    callbacks=[checkpoint_callback]  # Enable checkpointing
)


# In[42]:


nit_model.train()


# In[43]:


nit_model_lightining = NITModelLightning(nit_model)


# In[44]:


trainer.fit(nit_model_lightining, train_loader, test_loader)

save_model(nit_model_lightining.model, "CNNModelback.pth")

save_model(nit_model, "NITModelback.pth")
# # Testing Model

# In[39]:


# q = [X_test[0]]


# In[40]:


# q


# In[41]:


# original_outputs = nit_model.pipeline_outputs(q)
# altered_outputs = nit_model.pipeline_outputs(q,altered=True)


# In[47]:


# original_outputs.logits.mean()


# In[46]:


# altered_outputs.logits.mean()


# In[43]:


# decoded_original = nit_model.tokenizer.batch_decode(original_outputs.logits, skip_special_tokens=True)


# In[59]:


# original  = nit_model.altered_pipeline(q)


# In[60]:


# outputs= nit_model.llm.generate(
#             input_ids=None,
#             inputs_embeds=original.input_embeds,
#             attention_mask=original.attention_mask,
#             pad_token_id=nit_model.tokenizer.pad_token_id,  # Padding token ID
#             eos_token_id=nit_model.tokenizer.eos_token_id,  # End-of-sequence token ID
#             no_repeat_ngram_size=2,
#             return_dict=True
#         )


# In[61]:


# nit_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# In[62]:


# F.mse_loss(nit_model.altered_pipeline(q).input_embeds, nit_model.original_pipeline(q).input_embeds)


# In[65]:


# load_checkpoint(nit_model,optimizer)

