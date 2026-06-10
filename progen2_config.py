"""
Copy to: pretrained/progen2-small/configuration_progen2.py
"""
from transformers import PretrainedConfig

class ProGen2Config(PretrainedConfig):
    model_type = "progen2"
    def __init__(self, vocab_size=50400, n_positions=2048, n_embd=768, n_layer=12,
                 n_head=12, activation_function="gelu", resid_pdrop=0.1,
                 embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-5,
                 initializer_range=0.02, bos_token_id=0, eos_token_id=1, **kwargs):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
