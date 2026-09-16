"""TrueSkill ranking for forced-choice street-image comparisons."""

from __future__ import annotations

from typing import Iterable

import trueskill


DEFAULT_ENVIRONMENT = trueskill.TrueSkill(
    mu=25.0,
    sigma=25.0 / 3.0,
    beta=25.0 / 6.0,
    tau=25.0 / 300.0,
    draw_probability=0.0,
)


def calculate_trueskill(
    votes: Iterable[tuple[str, str, str]],
    env: trueskill.TrueSkill = DEFAULT_ENVIRONMENT,
) -> dict[str, dict[str, float]]:
    """Update TrueSkill ratings sequentially from forced-choice vote rows."""

    ratings: dict[str, trueskill.Rating] = {}
    for image_a, image_b, winner in votes:
        ratings.setdefault(image_a, env.create_rating())
        ratings.setdefault(image_b, env.create_rating())
        loser = image_b if winner == image_a else image_a
        winner_rating, loser_rating = env.rate_1vs1(ratings[winner], ratings[loser])
        ratings[winner] = winner_rating
        ratings[loser] = loser_rating

    return {
        image: {"trueskill_mu": rating.mu, "trueskill_sigma": rating.sigma}
        for image, rating in sorted(ratings.items())
    }
