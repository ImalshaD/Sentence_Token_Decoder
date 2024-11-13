#!/usr/bin/env python
# coding: utf-8

# In[43]:


from NitMultilingualEncoders import NitQwenMathInstruct, NitMT5encoder, NitRobertaencoder


# In[44]:


from AlignmentModels import Conv1dAutoencoder, CNN1DRBencoder, CNN1DRBdecoder, LSTMDecoder, TransformerDecoderModel


# In[45]:


import numpy as np
import pandas as pd


# In[46]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# In[47]:


from torch.nn.utils import clip_grad_norm_


# In[48]:


# torch.cuda.set_device(4)


# In[49]:


torch.set_float32_matmul_precision('high')


# In[50]:


from tqdm import tqdm


# In[51]:


from datasets import load_dataset


# In[52]:


import re
def extract_num_output(text):
    match = re.search(r'(?<=The answer is:\s).*$', text)
    if match:
        return match.group(0)
    return None


# In[53]:


data_dir = "data_cache"
model_dir = "model_cache"
math_ds = load_dataset("meta-math/MetaMathQA", cache_dir=data_dir)
sen_ds = load_dataset("sentence-transformers/wikipedia-en-sentences",cache_dir=data_dir)


# In[ ]:


math_df = math_ds['train'].to_pandas()
sen_df = sen_ds['train'].to_pandas()


# In[ ]:


math_df.head()


# In[ ]:


sen_df.head()


# In[ ]:


math_df['Numerical_output']= math_df['response'].apply(extract_num_output)


# In[ ]:


math_df.head()


# In[ ]:


math_df.isna().sum()


# In[ ]:


sen_df.isna().sum()


# # Data Loaders

# In[ ]:


batch_size = 512


# In[ ]:


math_train_query = math_df['query']
sen_train = sen_df['sentence']


# In[ ]:


math_array = math_train_query.to_numpy()
sen_array = sen_train.to_numpy()


# In[ ]:


math_X_train, math_X_test = train_test_split(math_array, test_size=0.2, random_state=42)
sen_X_train, sen_X_test = train_test_split(sen_array, test_size=0.2, random_state=42)


# In[ ]:


combined_X_train = np.concatenate((math_X_train, sen_X_train), axis=0)


# In[ ]:


math_train_loader = DataLoader(math_X_train, batch_size=batch_size, shuffle=True)
math_test_loader = DataLoader(math_X_test, batch_size=batch_size, shuffle=False)


# In[ ]:


sen_train_loader = DataLoader(sen_X_train, batch_size=batch_size, shuffle=True)
sen_test_loader = DataLoader(sen_X_test, batch_size=batch_size, shuffle=False)


# In[ ]:


combined_train_loader = DataLoader(combined_X_train, batch_size=batch_size, shuffle=True)


# # LLM and Encoder init

# In[ ]:


max_tokens = 100
padding = "max_length"


# In[ ]:


qwen_template = "<|im_start|>{text}<|im_end|>"


# In[ ]:


# gpu_ids = [0,1,2,3]


# In[ ]:


qwen = NitQwenMathInstruct(cache_dir=model_dir, max_tokens=max_tokens, padding=padding)
qwen.setTemplate(qwen_template)


# In[ ]:


rb = NitRobertaencoder(cache_dir=model_dir, max_tokens=max_tokens, padding=padding)


# In[ ]:


# rb.useDataParellel(gpu_ids)
# qwen.useDataParellel(gpu_ids)


# In[ ]:


rb_embedding_shape = rb.getEmbedding_shape()
qwen_embedding_shape = qwen.getEmbedding_shape()


# In[ ]:


rb_embedding_shape, qwen_embedding_shape


# # Alignment Model

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


align_model = CNN1DRBencoder(rb_embedding_shape, qwen_embedding_shape).to(device)
# align_model = TransformerDecoderModel().to(device)


# In[ ]:


def count_parameters(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params


# In[ ]:


# align_model = torch.nn.DataParallel(align_model, device_ids=gpu_ids)
align_model = TransformerDecoderModel().to(device)


# In[ ]:


count_parameters(align_model)


# # Testing NITModels with embeddings

# In[ ]:


test_texts = ["how many legs do 400 dogs have, if each dog has 4 legs?"]


# In[ ]:


qwen.applyTemplate(test_texts)


# In[ ]:


qwen.get_embeddings_from_text(test_texts).input_embeds


# In[ ]:


rb.applyFormating(test_texts)


# In[ ]:


rb_embs = rb.get_embeddings_from_text(test_texts, pooler_output=True).input_embeds


# In[ ]:


rb_embs.shape


# In[ ]:


rb_embs[0]


# # Training

# In[ ]:


def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, "checkpoints/"+path)


# In[ ]:


# define train loop
def train(model,encoder,llm ,train_loader, test_loader, optimizer, criterion ,epochs=10, scaler=None, clip_value=1.0):
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        best_val_loss = np.inf

        for batch in tqdm(train_loader, desc = f'epoch_{epoch+1}/{epochs}'):
            
            with torch.no_grad():
                encoder_embedding = encoder.get_embeddings_from_text(batch, pooler_output=True).input_embeds
                llm_embedding = llm.get_embeddings_from_text(batch).input_embeds

            
            # Use AMP for mixed precision if scaler is provided
            # with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            output = model(encoder_embedding)
            loss = criterion(output, llm_embedding)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Numerical instability detected in training. Skipping this batch.")
                continue

            # Backpropagation with optional scaler
            if scaler:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                
            train_loss += loss.item() * output.size(0)

        train_loss /= len(train_loader.dataset)
        

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation :"):
                encoder_embedding = encoder.get_embeddings_from_text(batch, pooler_output=True).input_embeds
                llm_embedding = llm.get_embeddings_from_text(batch).input_embeds

                output = model(encoder_embedding)
                loss = criterion(output, llm_embedding)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Numerical instability detected in validation. Skipping this batch.")
                    continue

                val_loss += loss.item() * output.size(0)
                
            val_loss /= len(test_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(epoch, model, optimizer, best_val_loss, "best_model_transformers.pth")
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
        with open("trans_logs.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}\n")


# In[ ]:


optimizer = torch.optim.Adam(align_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
# scaler = torch.amp.GradScaler("cuda")


# In[ ]:


# train(align_model, rb, qwen, sen_train_loader, sen_test_loader, optimizer, criterion, epochs=2, scaler=None)


# In[ ]:


train(align_model, rb, qwen, math_train_loader, math_test_loader, optimizer, criterion, epochs=100, scaler=None, clip_value=1.0)


# In[ ]:


def save_model(model, filename='model.pth'):
    torch.save(model.module.state_dict(), filename)
    print(f"Model saved to {filename}")


# In[ ]:


save_model(align_model, filename='qwen_rb_pooler_trans.pth')

