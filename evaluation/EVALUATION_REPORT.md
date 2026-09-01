# Evaluation Report

This report is generated directly from `run_all_evaluations.py` -- every number below comes from an actual run of the code, not from hand-written estimates. Regenerate it any time with:

```
python run_all_evaluations.py
```

## 1. Search Algorithms (BFS / DFS / UCS / Greedy / A*)

| Query | Algorithm | Nodes Expanded | Path Length | Cost (km) | Time (ms) |
|---|---|---|---|---|---|
| Home -> Laboratory_A | BFS | 8 | 4 | 12.687 | 0.0424 |
| Home -> Laboratory_A | DFS | 5 | 4 | 12.687 | 0.0344 |
| Home -> Laboratory_A | UCS | 16 | 4 | 12.687 | 0.0715 |
| Home -> Laboratory_A | Greedy | 5 | 4 | 12.687 | 0.0663 |
| Home -> Laboratory_A | A* | 8 | 4 | 12.687 | 0.0183 |
| Home -> Pharmacy_B | BFS | 10 | 4 | 13.107 | 0.0104 |
| Home -> Pharmacy_B | DFS | 13 | 4 | 13.107 | 0.0093 |
| Home -> Pharmacy_B | UCS | 17 | 4 | 13.107 | 0.013 |
| Home -> Pharmacy_B | Greedy | 5 | 4 | 13.107 | 0.0121 |
| Home -> Pharmacy_B | A* | 5 | 4 | 13.107 | 0.0102 |

**Finding:** A* consistently reaches the same optimal cost as UCS (both are guaranteed-optimal here) while expanding fewer or equal nodes, because its admissible heuristic (straight-line distance) focuses the search toward the goal instead of exploring uniformly in all directions.

## 2. CSP Appointment Scheduling

- Naive backtracking: **39 nodes expanded**, 0.2065 ms
- Smart backtracking (MRV + forward checking): **16 nodes expanded**, 0.5078 ms
- Node expansion reduced by **59.0%**
- Feasible appointment slots found: 10

## 3. Decision Tree -- Service Category Classification

- Train / test split: 450 / 150 samples
- **Accuracy: 92.0%**

Classification report (precision / recall / F1 per class):
```
precision    recall  f1-score   support

              Emergency       0.92      0.92      0.92        12
            General_OPD       0.95      0.99      0.97        70
Specialist_Consultation       0.96      0.85      0.90        27
             Urgent_OPD       0.84      0.93      0.88        28
            Vaccination       0.90      0.69      0.78        13

               accuracy                           0.92       150
              macro avg       0.91      0.88      0.89       150
           weighted avg       0.92      0.92      0.92       150
```

Confusion matrix (rows = true label, cols = predicted), label order ['Emergency', 'General_OPD', 'Specialist_Consultation', 'Urgent_OPD', 'Vaccination']:
```
[11, 1, 0, 0, 0]
[0, 69, 0, 1, 0]
[1, 2, 23, 0, 1]
[0, 1, 1, 26, 0]
[0, 0, 0, 4, 9]
```

Top features driving the decision:
- chronic_condition: 0.314
- fever_severity_score: 0.2401
- needs_vaccination: 0.2091

**Note:** accuracy is below 100% by design -- the training data includes ~5% injected label noise to simulate real-world ambiguity, so a perfect score would actually indicate overfitting, not a better model.

## 4. K-Means -- Patient Experience Segmentation

- k = 4, **silhouette score = 0.3026** (closer to 1 = better separated)

| Cluster | Label | Size | Avg Wait (min) | Avg Satisfaction |
|---|---|---|---|---|
| 0 | High-friction users | 104 | 39.07 | 3.23 |
| 1 | Highly satisfied users | 99 | 10.42 | 8.97 |
| 2 | Digitally underserved users | 109 | 32.23 | 4.94 |
| 3 | Long-wait users | 88 | 62.71 | 3.97 |

**Finding:** the four discovered clusters line up with the four experience patterns the roadmap anticipated (highly satisfied, long-wait, high-friction, digitally underserved) -- K-Means recovered this structure without being told the group labels in advance.

## 5. Supervisor Agent -- End-to-End Evaluation

- **Task completion rate: 100.0%** (found a feasible appointment)

| Scenario | Service | Urgency | Appointment Found | Route Cost (km) | Response Time (ms) | Trace Steps |
|---|---|---|---|---|---|---|
| My mother has had a high fever since yesterda... | Emergency | high | Yes | 11.812 | 1.79 | 6 |
| I'm 28 years old and need a vaccination appoi... | Vaccination | routine | Yes | 11.697 | 0.65 | 6 |
| My father is 70 and has a chronic condition, ... | Specialist_Consultation | routine | Yes | 5.99 | 0.93 | 5 |

**Finding:** the agent completes the full perceive-reason-plan-act loop in well under a second per request on this synthetic dataset, with every decision traceable to a specific step (NLU extraction, expert-system rule firing, CSP constraint satisfaction, or A* routing) -- there is no unexplained black-box step in the pipeline.
