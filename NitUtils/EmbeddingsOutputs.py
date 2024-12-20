class EmbeddingsOutputs:
    def __init__(self, input_embeds, attention_mask):
        self.input_embeds = input_embeds
        self.attention_mask = attention_mask
    
    def fromTensor(self, embedding_tensor, attention_mask):
        return EmbeddingsOutputs(embedding_tensor, attention_mask)