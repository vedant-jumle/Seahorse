"""Teacher-forced log-probability metrics (no generation-based judging)."""

import torch


def cont_logprobs(model, prompt_ids, cont_ids, device):
    """Log-probs [C, V] of the next-token distributions that predict `cont_ids`."""
    ids = torch.cat([prompt_ids.cpu(), cont_ids.cpu()])[None].to(device)
    logits = model(ids).logits[0].float()
    p, c = prompt_ids.shape[0], cont_ids.shape[0]
    return torch.log_softmax(logits[p - 1 : p - 1 + c], dim=-1)


def kl(lp_p, lp_q):
    """Mean over positions of KL(p || q), given log-prob tensors [C, V]."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1).mean().item()


def seq_logprob(model, prompt_ids, cont_ids, device):
    """Total log P(cont_ids | prompt_ids)."""
    lp = cont_logprobs(model, prompt_ids, cont_ids, device)
    return lp.gather(1, cont_ids.to(lp.device)[:, None]).sum().item()


def entropy(logits):
    """Entropy (nats) of each row's softmax distribution."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


def greedy(model, tok, prompt_ids, n, device):
    """Greedy continuation token ids (without the prompt)."""
    ids = prompt_ids[None].to(device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=n,
        do_sample=False,
        repetition_penalty=1.0,  # Qwen's generation_config sets 1.05; keep pure greedy
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    return out[0, ids.shape[1] :].cpu()
