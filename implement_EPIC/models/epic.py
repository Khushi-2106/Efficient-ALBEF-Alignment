import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple ITC head: linear classifier on token embeddings
class ITCHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 1)  

    def forward(self, token_embeds, mask=None):
        # token_embeds: [B, T, D]
        logits = self.classifier(token_embeds).squeeze(-1)  # [B, T]
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)  # optional
        probs = torch.sigmoid(logits)
        return logits, probs

# EPIC Loss: binary cross entropy on token-level labels
def epic_loss_fn(logits, labels, mask=None):
    """
    logits: [B, T] raw logits (before sigmoid)
    labels: [B, T] 0/1 (1=consistent, 0=inconsistent)
    mask:   [B, T] boolean indicating valid token positions
    """
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask].float()
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='mean')
    return loss

# Attention-based saliency (cheap)
def compute_text_token_saliency(cross_attentions):
    """
    cross_attentions: list or tensor of cross-attn matrices
    Expect shape: [num_layers, B, num_heads, T_text, T_image] or single [B, heads, T_text, T_image]
    Return: saliency_scores [B, T_text] normalized [0,1]
    """
    # If list, average them
    if isinstance(cross_attentions, list):
        att = torch.stack([a.detach() for a in cross_attentions], dim=0)  # [L, B, H, Tt, Ti]
        att = att.mean(dim=0)  # [B, H, Tt, Ti]
    else:
        att = cross_attentions  # [B, H, Tt, Ti]
    # outgoing attention mass per text token (sum over image tokens and heads)
    saliency = att.sum(dim=-1).sum(dim=1)  # [B, Tt]
    # normalize per-sample
    smin = saliency.min(dim=1, keepdim=True)[0]
    smax = saliency.max(dim=1, keepdim=True)[0]
    saliency_norm = (saliency - smin) / (smax - smin + 1e-9)
    return saliency_norm

# Top-K text token selection
def topk_tokens_from_saliency(saliency, k):
    # saliency: [B, T]
    k = min(k, saliency.size(1))
    vals, idx = torch.topk(saliency, k=k, dim=1)
    mask = torch.zeros_like(saliency, dtype=torch.bool)
    batch_idx = torch.arange(saliency.size(0), device=saliency.device).unsqueeze(1)
    mask[batch_idx, idx] = True
    return mask  # boolean mask of selected tokens

# Utility: generate inconsistent tokens (requires HF transformers)
# NOTE: keep LM small (gpt2-small) to limit compute. Sampling top_k/top_p for variety.
def sample_replacements(tokenizer, lm_model, input_texts, replace_indices, device='cpu', max_new_tokens=1):
    """
    input_texts: list[str] original sentences
    replace_indices: boolean mask [B, T] indicating which tokens to replace
    returns: list of tokenized texts with replacements at indicated indices (strings or token ids)
    """
    # naive approach: convert tokens to ids, replace each selected token by a sample from LM conditioned on prefix
    # For brevity we return a list of token ids we expect the pretrain dataloader to convert back.
    outs = []
    for i, txt in enumerate(input_texts):
        toks = tokenizer.tokenize(txt)
        toks_ids = tokenizer.encode(txt, add_special_tokens=False)
        for j, do_replace in enumerate(replace_indices[i].tolist()):
            if do_replace:
                # simple sampling: sample one token id from lm logits (use decoder with prefix)
                # NOTE: to keep code short, here we call lm_model.generate on the prefix up to position j.
                prefix = tokenizer.decode(toks_ids[:j], skip_special_tokens=True)
                inputs = tokenizer(prefix, return_tensors='pt').to(device)
                # generate 1 token
                gen = lm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95)
                new_ids = gen[0][-(max_new_tokens):].tolist()
                # place first generated token id in position j (coarse)
                toks_ids[j] = new_ids[0]
        outs.append(toks_ids)
    return outs
