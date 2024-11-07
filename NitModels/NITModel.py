import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# from einops import rearrange

from .NITConfig import NITConfig
from ..AlignmentModels import AlignmentModel
from ..NitUtils import TokenizerOutputs, EmbeddingsOutputs

import torch.nn as nn
import torch.nn.functional as F

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
            self.freeze_lm()
    
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