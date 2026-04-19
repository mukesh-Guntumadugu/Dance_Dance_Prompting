"""
onset_reward.py
===============
Reward function for GRPO RL training of onset detection.

Given a model's raw text output and a list of ground-truth onset
timestamps (in seconds), computes an F1-based reward in [0.0, 1.0].

A predicted onset is considered a true positive if it falls within
±TOLERANCE_SEC of any un-matched ground-truth onset.
"""

import re
from typing import List


TOLERANCE_SEC: float = 0.05   # ±50 ms window


def _parse_timestamps(text: str) -> List[float]:
    """
    Extract all numeric values from a string and treat them as seconds.
    Handles formats like:
        "1.23, 4.56, 7.89"
        "1234 ms, 4567 ms"              → converts ms → seconds
        "Hit : 1.23\nHit : 4.56"
        "0.5s, 1.0s"
    """
    # Convert explicit millisecond annotations first
    ms_matches = re.findall(r'(\d+(?:\.\d+)?)\s*ms', text)
    if ms_matches:
        values = [float(m) / 1000.0 for m in ms_matches]
        return sorted(set(values))

    # Otherwise extract all bare floats/ints
    raw = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    values = []
    for tok in raw:
        val = float(tok)
        # Heuristic: values > 600 are almost certainly milliseconds
        if val > 600.0:
            val /= 1000.0
        if 0.0 <= val <= 600.0:
            values.append(val)
    return sorted(set(values))


def onset_f1_reward(
    predicted_text: str,
    gt_onsets_sec: List[float],
    tolerance_sec: float = TOLERANCE_SEC,
) -> float:
    """
    Compute onset detection F1 score as a reward signal.

    Args:
        predicted_text: Raw text output from the model.
        gt_onsets_sec:  Ground-truth onset timestamps in seconds.
        tolerance_sec:  Match window in seconds (default ±50 ms).

    Returns:
        Float in [0.0, 1.0] — F1 score of model predictions vs ground truth.
        Returns 0.0 for empty predictions or empty ground truth.
    """
    predicted = _parse_timestamps(predicted_text)
    gt = list(gt_onsets_sec)

    if not predicted or not gt:
        return 0.0

    # Greedy matching: for each predicted onset, find the nearest unmatched GT
    gt_matched = [False] * len(gt)
    true_positives = 0

    for pred_t in predicted:
        best_dist = float("inf")
        best_idx = -1
        for i, gt_t in enumerate(gt):
            if gt_matched[i]:
                continue
            dist = abs(pred_t - gt_t)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0 and best_dist <= tolerance_sec:
            true_positives += 1
            gt_matched[best_idx] = True

    precision = true_positives / len(predicted) if predicted else 0.0
    recall    = true_positives / len(gt)        if gt        else 0.0

    if precision + recall == 0.0:
        return 0.0

    f1 = 2.0 * precision * recall / (precision + recall)
    return round(f1, 4)


# ── quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gt   = [0.5, 1.0, 1.5, 2.0, 2.5]
    pred = "0.52, 1.01, 1.55, 1.99, 2.48"  # all within ±50 ms → F1 = 1.0
    print(f"Perfect match  : {onset_f1_reward(pred, gt):.3f}")   # expect ~1.0

    pred2 = "0.52, 1.01"                    # only 2/5 GT matched
    print(f"Partial match  : {onset_f1_reward(pred2, gt):.3f}")  # expect ~0.5

    pred3 = "10.0, 20.0, 30.0"             # no matches
    print(f"No match       : {onset_f1_reward(pred3, gt):.3f}")  # expect 0.0
