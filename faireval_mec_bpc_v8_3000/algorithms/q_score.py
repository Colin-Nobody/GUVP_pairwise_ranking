"""Place Pulse style strength-of-schedule Q-score."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def calculate_q_scores(votes: Iterable[tuple[str, str, str]]) -> dict[str, dict[str, float]]:
    """Calculate the published Place Pulse style Q-score for each image.

    Each tuple is ``(image_a, image_b, winner)`` and winner must be one of the
    two images.  A win receives the mean win rate of the opponents it beat
    (award); a loss receives the mean loss rate of the opponents that beat it
    (penalty).  This is the strength-of-schedule correction used in Place
    Pulse follow-up work.
    """

    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    beaten_opponents: dict[str, list[str]] = defaultdict(list)
    losing_opponents: dict[str, list[str]] = defaultdict(list)
    images: set[str] = set()

    for image_a, image_b, winner in votes:
        images.update((image_a, image_b))
        loser = image_b if winner == image_a else image_a
        wins[winner] += 1
        losses[loser] += 1
        beaten_opponents[winner].append(loser)
        losing_opponents[loser].append(winner)

    total = {image: wins[image] + losses[image] for image in images}
    win_rate = {
        image: wins[image] / total[image] if total[image] else 0.0
        for image in images
    }
    loss_rate = {
        image: losses[image] / total[image] if total[image] else 0.0
        for image in images
    }

    results: dict[str, dict[str, float]] = {}
    for image in sorted(images):
        beaten = beaten_opponents[image]
        beat_by = losing_opponents[image]
        award = sum(win_rate[opponent] for opponent in beaten) / len(beaten) if beaten else 0.0
        penalty = sum(loss_rate[opponent] for opponent in beat_by) / len(beat_by) if beat_by else 0.0
        q_score = (10.0 / 3.0) * (win_rate[image] + award - penalty + 1.0)
        results[image] = {
            "wins": wins[image],
            "losses": losses[image],
            "comparison_count": total[image],
            "win_ratio": win_rate[image],
            "q_score_0_10": max(0.0, min(10.0, q_score)),
        }
    return results
