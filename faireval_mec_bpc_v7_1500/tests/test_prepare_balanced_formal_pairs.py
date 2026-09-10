from __future__ import annotations

import unittest
from collections import Counter

from scripts.prepare_balanced_formal_pairs import round_robin_rounds, select_balanced_pairs


class BalancedPairTests(unittest.TestCase):
    def test_1500_pairs_give_every_image_12_appearances(self) -> None:
        images = [f"image-{index:03d}.jpg" for index in range(250)]
        pairs = select_balanced_pairs(images, 1500, seed=20260827)
        participation = Counter(image for pair in pairs for image in pair)

        self.assertEqual(len(pairs), 1500)
        self.assertEqual(len({frozenset(pair) for pair in pairs}), 1500)
        self.assertEqual(set(participation.values()), {12})

    def test_round_robin_contains_every_unordered_pair_once(self) -> None:
        images = [f"image-{index}" for index in range(8)]
        pairs = [pair for round_pairs in round_robin_rounds(images) for pair in round_pairs]
        self.assertEqual(len(pairs), 28)
        self.assertEqual(len({frozenset(pair) for pair in pairs}), 28)

    def test_3000_pairs_give_every_image_24_appearances(self) -> None:
        images = [f"image-{index:03d}.jpg" for index in range(250)]
        pairs = select_balanced_pairs(images, 3000, seed=20260827)
        degrees = {image: 0 for image in images}
        for image_a, image_b in pairs:
            degrees[image_a] += 1
            degrees[image_b] += 1
        self.assertEqual(len(pairs), 3000)
        self.assertEqual(len({frozenset(pair) for pair in pairs}), 3000)
        self.assertEqual(set(degrees.values()), {24})

    def test_selection_is_reproducible(self) -> None:
        images = [f"image-{index}" for index in range(10)]
        self.assertEqual(
            select_balanced_pairs(images, 17, seed=7),
            select_balanced_pairs(images, 17, seed=7),
        )


if __name__ == "__main__":
    unittest.main()
