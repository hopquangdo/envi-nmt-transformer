import torch
import torch.nn.functional as F


@torch.inference_mode()
def greedy_decode(model, enc_out, src_mask, bos_id, eos_id, max_len, device):
    tgt = torch.tensor([[bos_id]], device=device)

    for _ in range(max_len):
        tgt_mask = torch.tril(
            torch.ones(tgt.size(1), tgt.size(1), device=device)
        ).bool().unsqueeze(0).unsqueeze(0)

        logits = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        next_token = logits[:, -1, :].argmax(-1).unsqueeze(1)

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    return tgt.squeeze(0).tolist()


@torch.inference_mode()
def beam_search_decode(
        model,
        enc_out,
        src_mask,
        bos_id,
        eos_id,
        max_len,
        beam_size,
        device
):
    enc_out = enc_out.repeat(beam_size, 1, 1)
    src_mask = src_mask.repeat(beam_size, 1, 1, 1)

    beams = torch.full((beam_size, 1), bos_id, device=device)
    scores = torch.zeros(beam_size, device=device)

    for _ in range(max_len):
        tgt_mask = torch.tril(
            torch.ones(beams.size(1), beams.size(1), device=device)
        ).bool().unsqueeze(0).unsqueeze(0)

        logits = model.decoder(beams, enc_out, src_mask, tgt_mask)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        total_scores = scores.unsqueeze(1) + log_probs

        topk_scores, topk_ids = total_scores.view(-1).topk(beam_size)

        beam_idx = topk_ids // log_probs.size(-1)
        token_idx = topk_ids % log_probs.size(-1)

        beams = torch.cat([
            beams[beam_idx],
            token_idx.unsqueeze(1)
        ], dim=1)

        scores = topk_scores

        if (token_idx == eos_id).all():
            break

    best = beams[scores.argmax()]
    return best.tolist()
