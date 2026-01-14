from core.flash_attn import flash_attention_forward

def flash_attention(q, k, v, scale=None):
    """Flash attention: softmax(Q @ K.T / sqrt(d)) @ V"""
    return flash_attention_forward(q.contiguous(), k.contiguous(), v.contiguous())
