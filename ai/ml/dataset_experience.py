"""
Synthetic Dataset -- Patient Experience Segmentation
========================================================

Generates synthetic post-visit experience records for K-Means
clustering. This is NOT about diagnosis or service type -- it's about
how the NAVIGATION EXPERIENCE felt, which is what the roadmap's
"Experience Optimization" module (Section 13) is meant to analyze.

Features (all post-visit metrics):
    waiting_time_minutes     how long they waited
    navigation_steps         how many steps/screens/questions it took
                              to get to the right service
    travel_distance_km       how far they had to travel
    digital_usage_score      0 (did everything in person/by phone) -
                              10 (fully self-served via app/web)
    satisfaction_score       1 (very unhappy) - 10 (very happy)
    visit_count               how many times they've used the system before

Four underlying (synthetic) population groups are baked in so K-Means
has real structure to discover -- mirroring the roadmap's expected
clusters: highly satisfied, long-wait, high-friction, digitally
underserved.
"""

import random


def generate_dataset(n_samples: int = 400, seed: int = 7) -> list:
    rng = random.Random(seed)
    rows = []

    # Each "profile" is a rough center + spread for the 6 features.
    # These are DELIBERATELY not labeled with the group name in the
    # output -- K-Means has to discover the grouping on its own, we
    # only use these to generate realistic, separable synthetic data.
    profiles = [
        # (waiting, steps, distance, digital_usage, satisfaction, visits)
        {"wait": (10, 5), "steps": (3, 1), "dist": (4, 2), "digital": (8, 1.5), "sat": (9, 0.7), "visits": (3, 2)},   # highly satisfied
        {"wait": (60, 15), "steps": (6, 2), "dist": (8, 3), "digital": (5, 2), "sat": (4, 1), "visits": (2, 1.5)},    # long-wait
        {"wait": (40, 12), "steps": (9, 2), "dist": (10, 3), "digital": (4, 2), "sat": (3, 1), "visits": (1, 1)},     # high-friction
        {"wait": (30, 10), "steps": (7, 2), "dist": (6, 3), "digital": (1, 1), "sat": (5, 1.5), "visits": (4, 2)},    # digitally underserved
    ]

    for _ in range(n_samples):
        profile = rng.choice(profiles)

        def sample(mean_std):
            mean, std = mean_std
            return max(0, rng.gauss(mean, std))

        rows.append({
            "waiting_time_minutes": round(sample(profile["wait"]), 1),
            "navigation_steps": max(1, round(sample(profile["steps"]))),
            "travel_distance_km": round(sample(profile["dist"]), 1),
            "digital_usage_score": round(min(10, sample(profile["digital"])), 1),
            "satisfaction_score": round(min(10, max(1, sample(profile["sat"]))), 1),
            "visit_count": max(0, round(sample(profile["visits"]))),
        })

    return rows


FEATURE_NAMES = [
    "waiting_time_minutes", "navigation_steps", "travel_distance_km",
    "digital_usage_score", "satisfaction_score", "visit_count",
]


if __name__ == "__main__":
    data = generate_dataset(10)
    for row in data:
        print(row)
