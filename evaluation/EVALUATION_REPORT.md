# Evaluation Report

This report is generated directly from `run_all_evaluations.py` -- every number below comes from an actual run of the code, not from hand-written estimates. Regenerate it any time with:

```
python run_all_evaluations.py
```

## 1. Search Algorithms (BFS / DFS / UCS / Greedy / A*)

| Query | Algorithm | Nodes Expanded | Path Length | Cost (km) | Time (ms) |
|---|---|---|---|---|---|
| Home -> Laboratory_A | BFS | 10 | 4 | 12.687 | 0.0146 |
| Home -> Laboratory_A | DFS | 5 | 4 | 12.687 | 0.0332 |
| Home -> Laboratory_A | UCS | 18 | 4 | 12.687 | 0.0179 |
| Home -> Laboratory_A | Greedy | 5 | 4 | 12.687 | 0.0152 |
| Home -> Laboratory_A | A* | 5 | 4 | 12.687 | 0.012 |
| Home -> Pharmacy_B | BFS | 12 | 4 | 13.107 | 0.0077 |
| Home -> Pharmacy_B | DFS | 15 | 7 | 32.954 | 0.0086 |
| Home -> Pharmacy_B | UCS | 19 | 4 | 13.107 | 0.0103 |
| Home -> Pharmacy_B | Greedy | 5 | 4 | 13.107 | 0.0118 |
| Home -> Pharmacy_B | A* | 5 | 4 | 13.107 | 0.0102 |

**Finding:** UCS and A* are the distance-optimal algorithms on this non-negative weighted graph. A* uses the straight-line heuristic to focus the search; measured node counts and runtime are reported for the actual run rather than treated as universal complexity claims.

## 2. AI Route Optimization (K-shortest paths + algorithm benchmark)

| Query | Candidate routes | Selected route | Distance (km) | Est. time (min) |
|---|---:|---:|---:|---:|
| Home -> Laboratory_A | 3 | Route 2 | 12.761 | 21.83 |
| Home -> Pharmacy_B | 3 | Route 1 | 13.107 | 21.39 |
| Home -> OPD_C | 3 | Route 1 | 5.233 | 11.65 |

The route engine generates multiple loopless alternatives using Yen's K-shortest-path method and selects one according to an explicit objective. The same query is benchmarked with BFS, DFS, UCS, Greedy Best-First, and A* so the project can compare route quality, nodes expanded, runtime, and theoretical complexity.

## 3. Multi-Vehicle Optimization (Genetic Algorithm)

- Vehicles: 2
- Total optimized distance: **41.614 km**
- Maximum vehicle distance: **21.54 km**
- Route balance gap: **1.466 km**
- Fitness complexity: `O(G × P × S), where G=generations, P=population size, S=stops`

## 4. CSP Appointment Scheduling

- Naive backtracking: **39 nodes expanded**, 0.1085 ms
- Smart backtracking (MRV + forward checking): **16 nodes expanded**, 0.3501 ms
- Node expansion reduced by **59.0%**
- Feasible appointment slots found: 10

## 5. Decision Tree -- Service Category Classification

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

## 6. K-Means -- Patient Experience Segmentation

- k = 4, **silhouette score = 0.3026** (closer to 1 = better separated)

| Cluster | Label | Size | Avg Wait (min) | Avg Satisfaction |
|---|---|---|---|---|
| 0 | High-friction users | 104 | 39.07 | 3.23 |
| 1 | Highly satisfied users | 99 | 10.42 | 8.97 |
| 2 | Digitally underserved users | 109 | 32.23 | 4.94 |
| 3 | Long-wait users | 88 | 62.71 | 3.97 |

**Finding:** the four discovered clusters line up with the four experience patterns the roadmap anticipated (highly satisfied, long-wait, high-friction, digitally underserved) -- K-Means recovered this structure without being told the group labels in advance.

## 7. Supervisor Agent -- End-to-End Evaluation

- **Task completion rate: 100.0%** (found a feasible appointment)

| Scenario | Service | Urgency | Appointment Found | Route Cost (km) | Response Time (ms) | Trace Steps |
|---|---|---|---|---|---|---|
| My mother has had a high fever since yesterda... | Emergency | high | Yes | 11.886 | 1.38 | 6 |
| I'm 28 years old and need a vaccination appoi... | Vaccination | routine | Yes | 11.771 | 0.57 | 6 |
| My father is 70 and has a chronic condition, ... | Specialist_Consultation | routine | Yes | 5.233 | 0.8 | 5 |

**Finding:** the agent completes the full perceive-reason-plan-act loop in well under a second per request on this synthetic dataset, with every decision traceable to a specific step (NLU extraction, expert-system rule firing, CSP constraint satisfaction, or A* routing) -- there is no unexplained black-box step in the pipeline.
