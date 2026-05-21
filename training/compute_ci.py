import csv
import math
import os
import argparse
from typing import List, Tuple


DEFAULT_ALPHA = 0.001  # default 0.1% FPR
EPS = 1e-6             # small value to avoid log(0)
BLOCK_SIZE = 8640


def load_rows(csv_path: str) -> List[List[str]]:
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        header = next(r)  # skip header
        return list(r)


def compute_threshold(scores_real: List[float], alpha: float) -> float:
    # Choose threshold so that fraction of real >= theta ~ alpha
    n = len(scores_real)
    if n == 0:
        return float("inf")
    k = int(math.ceil(alpha * n))  # expected real counted as "fake" above threshold
    if k <= 0:
        # aim FPR ~ 0: pick +inf threshold
        return float("inf")
    # sort ascending; we want the (n - k + 1)-th smallest as threshold
    s = sorted(scores_real)
    idx = max(0, n - k)  # 0-based index
    if idx >= n:
        idx = n - 1
    return s[idx]


def compute_block_ci(block_rows: List[List[str]]) -> Tuple[int, int, float, float, float]:
    # Extract scores and labels
    real_scores: List[float] = []
    fake_scores: List[float] = []
    for row in block_rows:
        # Columns: filename, label, logit_real, logit_fake, prob_fake
        label = int(row[1])
        try:
            score = float(row[4])  # prob_fake
        except Exception:
            # fallback to logit_fake if prob is missing
            score = float(row[3])
        if label == 0:
            real_scores.append(score)
        else:
            fake_scores.append(score)

    n_real = len(real_scores)
    n_fake = len(fake_scores)

    theta = compute_threshold(real_scores, args.alpha)
    # ADR: mean over fake of [score >= theta]
    if n_fake == 0:
        adr = 0.0
    else:
        c = sum(1 for s in fake_scores if s >= theta)
        adr = c / n_fake

    # CI = -log10(max(ADR, eps))
    from math import log10
    ci = -log10(max(adr, EPS))
    return n_real, n_fake, theta, adr, ci


def list_method_names(dir_path: str) -> List[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    names = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default="logits_degrad.csv")
    parser.add_argument("--out_csv", type=str, default="ci_results_degrad.csv")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="FPR (e.g., 0.001 for 0.1%)")
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--method_names_dir", type=str, default="", help="Directory containing 28 method subfolders to name rows")
    global args
    args = parser.parse_args()
    in_csv = args.in_csv
    out_csv = args.out_csv
    rows = load_rows(in_csv)
    total = len(rows)
    if total % args.block_size != 0:
        print(f"[Warn] data rows ({total}) not divisible by block size {args.block_size}.")
    n_blocks = total // args.block_size
    rem = total % BLOCK_SIZE
    if n_blocks == 0 and rem == 0:
        print("[Error] no blocks found.")
        return

    results = []
    for b in range(n_blocks):
        start = b * args.block_size
        end = start + args.block_size
        block = rows[start:end]
        n_real, n_fake, theta, adr, ci = compute_block_ci(block)
        results.append((b + 1, n_real, n_fake, theta, adr, ci))

    # include last partial block if present
    if rem > 0:
        start = n_blocks * args.block_size
        end = total
        block = rows[start:end]
        b = n_blocks  # zero-based index for partial
        n_real, n_fake, theta, adr, ci = compute_block_ci(block)
        results.append((b + 1, n_real, n_fake, theta, adr, ci))

    # write csv
    # attach method names if provided
    names = list_method_names(args.method_names_dir)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        theta_hdr = f"theta@FPR={args.alpha*100:.1f}%"
        if names:
            w.writerow(["method_index(1-based)", "method_name", "n_real", "n_fake", theta_hdr, "ADR", "CI"])
            N = min(len(results), len(names))
            for i in range(N):
                idx, n_real, n_fake, theta, adr, ci = results[i]
                w.writerow([idx, names[i], n_real, n_fake, theta, adr, ci])
            for i in range(N, len(results)):
                idx, n_real, n_fake, theta, adr, ci = results[i]
                w.writerow([idx, "", n_real, n_fake, theta, adr, ci])
        else:
            w.writerow(["method_index(1-based)", "n_real", "n_fake", theta_hdr, "ADR", "CI"])
            for r in results:
                w.writerow(r)

    # print brief summary
    print(f"[Done] Computed CI for {len(results)} methods (blocks) at FPR={args.alpha*100:.1f}%.")
    print(f"       Wrote: {out_csv}")


if __name__ == "__main__":
    main()
