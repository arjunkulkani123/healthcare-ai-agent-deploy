"""
Synthetic Dataset -- Service Classification
===============================================

Generates a realistic-looking (but fully synthetic, no real patient
data) dataset for training a Decision Tree that predicts which SERVICE
CATEGORY a patient should be routed to.

Features (inputs):
    age                     numeric
    fever_severity_score    0 (none) - 3 (high)
    fever_duration_days     numeric
    chronic_condition       0/1
    needs_vaccination       0/1
    mobility_impaired       0/1
    prior_visits_count      numeric (how many times they've visited before)

Label (output) -- SERVICE CATEGORY, not a diagnosis:
    Emergency, Urgent_OPD, Vaccination, Specialist_Consultation, General_OPD

The generation logic deliberately mirrors (but does not call) the
expert-system rules in ai/expert_system/, so the Decision Tree is
learning to approximate the same kind of decision boundary a rule-based
system encodes explicitly -- a good talking point for the viva: "the
expert system is explicit logic; the decision tree learns similar
patterns statistically, with noise added to mimic real-world messiness."
"""

import random


def generate_dataset(n_samples: int = 600, seed: int = 42) -> list:
    rng = random.Random(seed)
    rows = []

    for _ in range(n_samples):
        age = rng.randint(1, 90)
        fever_severity_score = rng.choice([0, 0, 0, 1, 1, 2, 3])  # skewed toward "none/mild"
        fever_duration_days = rng.randint(0, 5) if fever_severity_score > 0 else 0
        chronic_condition = rng.choice([0, 0, 0, 1])
        needs_vaccination = rng.choice([0, 0, 0, 1]) if fever_severity_score == 0 else 0
        mobility_impaired = rng.choice([0, 0, 0, 0, 1])
        prior_visits_count = rng.randint(0, 12)

        label = _true_label(
            age, fever_severity_score, fever_duration_days,
            chronic_condition, needs_vaccination,
        )

        # Inject a little label noise (~5%) to simulate real-world
        # messiness (e.g. borderline/ambiguous cases) -- without this,
        # the tree trivially gets 100% accuracy, which is unrealistic
        # and a red flag in any evaluation section.
        if rng.random() < 0.05:
            label = rng.choice(
                ["Emergency", "Urgent_OPD", "Vaccination", "Specialist_Consultation", "General_OPD"]
            )

        rows.append({
            "age": age,
            "fever_severity_score": fever_severity_score,
            "fever_duration_days": fever_duration_days,
            "chronic_condition": chronic_condition,
            "needs_vaccination": needs_vaccination,
            "mobility_impaired": mobility_impaired,
            "prior_visits_count": prior_visits_count,
            "service_category": label,
        })

    return rows


def _true_label(age, fever_severity_score, fever_duration_days, chronic_condition, needs_vaccination) -> str:
    senior = age >= 60
    requires_medical_evaluation = fever_severity_score >= 2 and fever_duration_days >= 1

    if requires_medical_evaluation and senior:
        return "Emergency"
    if requires_medical_evaluation and not senior:
        return "Urgent_OPD"
    if needs_vaccination:
        return "Vaccination"
    if chronic_condition:
        return "Specialist_Consultation"
    return "General_OPD"


FEATURE_NAMES = [
    "age", "fever_severity_score", "fever_duration_days",
    "chronic_condition", "needs_vaccination", "mobility_impaired",
    "prior_visits_count",
]
LABEL_NAME = "service_category"


if __name__ == "__main__":
    data = generate_dataset(10)
    for row in data:
        print(row)
