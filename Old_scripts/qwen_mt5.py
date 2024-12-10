#!/usr/bin/env python
# coding: utf-8

# In[1]:


from NitMultilingualEncoders import NitQwenMathInstruct, NitMT5encoder, NitRobertaencoder


# In[2]:


from AlignmentModels import Conv1dAutoencoder, CNN1DRBencoder, CNN1DRBdecoder, LSTMDecoder, TransformerDecoderModel


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# In[5]:


from torch.nn.utils import clip_grad_norm_


# In[6]:


# torch.cuda.set_device(4)


# In[7]:


torch.set_float32_matmul_precision('high')


# In[8]:


from tqdm import tqdm


# In[9]:


from datasets import load_dataset


# In[10]:


import re
def extract_num_output(text):
    match = re.search(r'(?<=The answer is:\s).*$', text)
    if match:
        return match.group(0)
    return None


# In[11]:


data_dir = "data_cache"
model_dir = "model_cache"
math_ds = load_dataset("meta-math/MetaMathQA", cache_dir=data_dir)
sen_ds = load_dataset("sentence-transformers/wikipedia-en-sentences",cache_dir=data_dir)


# In[12]:


math_df = math_ds['train'].to_pandas()
sen_df = sen_ds['train'].to_pandas()


# In[13]:


math_df.head()


# In[14]:


sen_df.head()


# In[15]:


math_df['Numerical_output']= math_df['response'].apply(extract_num_output)


# In[16]:


math_df.head()


# In[17]:


math_df.isna().sum()


# In[18]:


sen_df.isna().sum()


# # Data Loaders

# In[19]:


batch_size = 1024


# In[20]:


math_train_query = math_df['query']
sen_train = sen_df['sentence']


# In[21]:


math_array = math_train_query.to_numpy()
sen_array = sen_train.to_numpy()


# In[22]:


math_X_train, math_X_test = train_test_split(math_array, test_size=0.2, random_state=42)
sen_X_train, sen_X_test = train_test_split(sen_array, test_size=0.2, random_state=42)


# In[23]:


combined_X_train = np.concatenate((math_X_train, sen_X_train), axis=0)


# In[24]:


math_train_loader = DataLoader(math_X_train, batch_size=batch_size, shuffle=True)
math_test_loader = DataLoader(math_X_test, batch_size=batch_size, shuffle=False)


# In[25]:


sen_train_loader = DataLoader(sen_X_train, batch_size=batch_size, shuffle=True)
sen_test_loader = DataLoader(sen_X_test, batch_size=batch_size, shuffle=False)


# In[26]:


combined_train_loader = DataLoader(combined_X_train, batch_size=batch_size, shuffle=True)


# # LLM and Encoder init

# In[27]:


max_tokens = 100
padding = "max_length"


# In[28]:


qwen_template = "<|im_start|>{text}<|im_end|>"


# In[29]:


# gpu_ids = [0,1,2,3]


# In[30]:


qwen = NitQwenMathInstruct(cache_dir=model_dir, max_tokens=max_tokens, padding=padding)
qwen.setTemplate(qwen_template)


# In[31]:


rb = NitRobertaencoder(cache_dir=model_dir, max_tokens=max_tokens, padding=padding)


# In[32]:


# rb.useDataParellel(gpu_ids)
# qwen.useDataParellel(gpu_ids)


# In[33]:


rb_embedding_shape = rb.getEmbedding_shape()
qwen_embedding_shape = qwen.getEmbedding_shape()


# In[34]:


rb_embedding_shape, qwen_embedding_shape


# # Alignment Model

# In[35]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[36]:


align_model = CNN1DRBencoder(rb_embedding_shape, qwen_embedding_shape).to(device)
# align_model = TransformerDecoderModel().to(device)


# In[37]:


# align_model = torch.nn.DataParallel(align_model, device_ids=gpu_ids)
align_model = CNN1DRBdecoder(rb_embedding_shape, qwen_embedding_shape).to(device)


# # Testing NITModels with embeddings

# In[38]:


test_texts = ["how many legs do 400 dogs have, if each dog has 4 legs?"]


# In[39]:


qwen.applyTemplate(test_texts)


# In[40]:


qwen.get_embeddings_from_text(test_texts).input_embeds


# In[41]:


rb.applyFormating(test_texts)


# In[42]:


rb_embs = rb.get_embeddings_from_text(test_texts, pooler_output=True).input_embeds


# In[43]:


rb_embs.shape


# In[44]:


rb_embs[0]


# # Training

# In[45]:


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
        model_saved_at_epoch = False

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
                save_checkpoint(epoch, model, optimizer, best_val_loss, "best_model.pth")
                print(f"Best model saved with loss: {best_val_loss:.7f} at epoch {epoch+1}/{epochs}")
                model_saved_at_epoch = True
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
        with open("logs.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f} model saved {model_saved_at_epoch}\n")
            model_saved_at_epoch = False


# In[47]:


optimizer = torch.optim.Adam(align_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
# scaler = torch.amp.GradScaler("cuda")


# In[48]:


# train(align_model, rb, qwen, sen_train_loader, sen_test_loader, optimizer, criterion, epochs=2, scaler=None)


# In[49]:


train(align_model, rb, qwen, math_train_loader, math_test_loader, optimizer, criterion, epochs=100, scaler=None, clip_value=1.0)


# In[ ]:


def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


# In[ ]:


save_model(align_model, filename='qwen_rb.pth')

