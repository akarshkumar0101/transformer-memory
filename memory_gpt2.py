from transformers import GPT2LMHeadModel, GPT2Model, AutoTokenizer, AutoModelForCausalLM

class MemoryGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ...
        self.init_weights()

    def forward(self, input_ids, ):
        super().forward(input_ids)

class MemoryGPT2Model(GPT2Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ...
        self.init_weights()
        
    def forward(self, input_ids):
        super().forward(input_ids)