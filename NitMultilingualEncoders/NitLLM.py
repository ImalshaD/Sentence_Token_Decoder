from ..NitUtils import EmbeddingsOutputs
from .NitEncoder import NitEncoder
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM

class NitLLM(NitEncoder, ABC):
    
    def __init__(self,llm_name,cache_dir):
        model = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir = cache_dir,padding_side='left')
        super().__init__(model, tokenizer)

    def get_outputs(self, embeddings : EmbeddingsOutputs):
        embeddings = self.get_embeddings
        return self.model(
            attention_mask=embeddings.attention_mask,
            inputs_embeds=embeddings.input_embeds,
            return_dict=True
        )
    
    def get_original_outputs(self, texts):
        return self.get_outputs(self.original_pipeline(texts))
    
    @abstractmethod
    def get_outputlogits(self, outputs):
        pass
