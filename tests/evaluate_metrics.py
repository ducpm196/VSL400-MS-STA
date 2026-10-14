import sys

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def calculate_n_gram_precision(reference: str, hypothesis: str, n: int = 4) -> float:
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    if len(hyp_words) < n:
        return 0.0
    ref_ngrams = [" ".join(ref_words[i : i + n]) for i in range(len(ref_words) - n + 1)]
    hyp_ngrams = [" ".join(hyp_words[i : i + n]) for i in range(len(hyp_words) - n + 1)]
    if not hyp_ngrams:
        return 0.0
    return sum(1 for g in hyp_ngrams if g in ref_ngrams) / len(hyp_ngrams)


def calculate_bleu4(reference: str, hypothesis: str) -> float:
    ps = [calculate_n_gram_precision(reference, hypothesis, n) for n in range(1, 5)]
    if min(ps) <= 0:
        return 0.0
    return float(np.exp(0.25 * sum(np.log(p) for p in ps)))


def calculate_rouge_l(reference: str, hypothesis: str) -> float:
    ref = reference.strip().lower().split()
    hyp = hypothesis.strip().lower().split()
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0.0
    rec = lcs / m if m else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def evaluate_signer_independent_split(predictions_list: list[dict]) -> tuple[float, float]:
    total_bleu, total_rouge = 0.0, 0.0
    n = len(predictions_list)
    print(f"[*] Kiểm thử Signer-Independent trên {n} mẫu...")
    for sample in predictions_list:
        total_bleu += calculate_bleu4(sample["ref_text"], sample["pred_text"])
        total_rouge += calculate_rouge_l(sample["ref_text"], sample["pred_text"])
    avg_bleu = (total_bleu / n) * 100
    avg_rouge = (total_rouge / n) * 100
    print("=" * 50)
    print(f"BLEU-4:   {avg_bleu:.2f}%  (Cam kết: > 25%)")
    print(f"ROUGE-L:  {avg_rouge:.2f}%  (Cam kết: > 45%)")
    print("=" * 50)
    return avg_bleu, avg_rouge


if __name__ == "__main__":
    mock_test = [
        {"ref_text": "Hôm nay tôi đi đến trường học.", "pred_text": "Hôm nay tôi đi đến trường học."},
        {"ref_text": "Bệnh viện nằm ở đâu vậy.", "pred_text": "Bệnh viện nằm ở đâu."},
        {"ref_text": "Tôi đang bị đau bụng thưa bác sĩ.", "pred_text": "Tôi bị đau bụng thưa bác sĩ."},
    ]
    evaluate_signer_independent_split(mock_test)
