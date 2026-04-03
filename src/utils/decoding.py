import torch
import torch.nn.functional as F


@torch.inference_mode()
def greedy_decode(model, enc_out, src_mask, bos_id, eos_id, max_len, device):
    """
    Dịch theo thuật toán Greedy (chọn từ có xác suất cao nhất tại mỗi bước).

    Input Demo:
        enc_out: Tensor (1, T_src, d_model) -> (1, 20, 512)
        src_mask: Tensor (1, 1, 1, T_src)
        bos_id: 2 (ID của <bos>)
        max_len: 50
    Output Demo:
        return: list[int] -> ví dụ: [2, 5, 10, 3] (Danh sách ID từ đã dịch)
    """
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
    """
    Dịch theo thuật toán Beam Search (giữ lại K kết quả tốt nhất tại mỗi bước).

    Input Demo:
        enc_out: Tensor (1, T_src, d_model)
        beam_size: 5
    Output Demo:
        return: list[int] -> Danh sách ID từ của kết quả tốt nhất.
    """
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
