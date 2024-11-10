from NitUtils import EmbeddingsOutputs
from .NitEncoder import NitEncoder
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM

class NitLLM(NitEncoder, ABC):
    
    def __init__(self,llm_name,cache_dir, max_tokens = 100 , padding='max_length'):
        model = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir = cache_dir,padding_side='right')
        super().__init__(model, tokenizer, max_tokens, padding)

    def get_outputs(self, embeddings : EmbeddingsOutputs, **kwargs):
        return self.model(
            attention_mask=embeddings.attention_mask,
            inputs_embeds=embeddings.input_embeds,
            return_dict=True,
            **kwargs
        )
    
    def get_generation(self, embeddings : EmbeddingsOutputs, **kwargs):
        outputs = self.model.generate(
            attention_mask=embeddings.attention_mask,
            inputs_embeds=embeddings.input_embeds,
            **kwargs
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    def get_outputs_from_text(self, texts, **kwargs):
        return self.get_outputs(self.get_embeddings_from_text(texts), **kwargs)
    
    @abstractmethod
    def get_outputlogits(self, texts,**kwargs):
        pass

    @abstractmethod
    def get_outputlogits_from_embeddings(self, embeddings : EmbeddingsOutputs, **kwargs):
        pass

class NitQwenMathInstruct(NitLLM):
    def __init__(self,cache_dir, max_tokens=100, padding = 'max_length' ,use_best = True):
        super().__init__("Qwen/Qwen2.5-Math-1.5B-Instruct",cache_dir,max_tokens,padding)
        self.best_config = {
            'max_new_tokens' : 512,
            'do_sample' : True, 
            'temperature' : 0.3
        }
        self.use_best = use_best
    
    def setUseBest(self,use_best):
        self.use_best = use_best
    
    def get_outputlogits(self, texts, **kwargs):
        if self.use_best:
            return self.get_outputs_from_text(texts,**self.best_config).logits
        return self.get_outputs_from_text(texts,**kwargs).logits
    
    def get_outputlogits_from_embeddings(self, embeddings : EmbeddingsOutputs, **kwargs):
        if self.use_best:
            return super().get_outputs(embeddings, **self.best_config).logits
        return super().get_outputs(embeddings, **kwargs).logits

    def get_generation(self, embeddings : EmbeddingsOutputs, **kwargs):
        if self.use_best:
            return super().get_generation(embeddings,**self.best_config)
        return super().get_generation(embeddings,**kwargs)
