from transformers import PretrainedConfig

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