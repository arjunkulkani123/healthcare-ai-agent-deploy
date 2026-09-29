# Healthcare Experience Optimization Agent

An agentic AI system that helps citizens navigate public healthcare
services — finding the right facility, checking eligibility, scheduling
around real-world constraints, and explaining every recommendation —
built to demonstrate CO1–CO4 of the AI course syllabus (agents, search,
knowledge/expert systems, ML) plus an image-processing component.

**This is not a diagnosis tool.** It optimizes access → navigation →
service selection → scheduling → information → experience → feedback.

## Status

- [x] Step 1 — Problem statement, users, use cases, PEAS (`docs/01_problem_and_peas.md`)
- [x] Step 2 — Search: BFS / DFS / UCS / Greedy / A* (`ai/search/`)
- [x] Step 3 — CSP appointment optimizer (`ai/csp/`)
- [x] Step 4 — Knowledge base + forward/backward chaining (`ai/knowledge/`)
- [x] Step 5 — Expert system + explanation engine (`ai/expert_system/`)
- [x] Step 6 — Decision Tree service classifier + K-Means segmentation (`ai/ml/`)
- [x] Step 7 — Document/image understanding (OCR) (`ai/vision/`)
- [x] Step 8 — Supervisor agent + tool orchestration (`agents/`)
- [x] Step 9 — UI (Streamlit) + agent trace panel (`app/`)
- [x] Step 10 — Evaluation metrics (`evaluation/`)
- [x] Step 11 — AI route optimization: K-shortest alternatives + algorithm benchmarking (`ai/optimization/routing/`)
- [x] Step 12 — Multi-vehicle route optimization using a Genetic Algorithm (`ai/optimization/multi_vehicle/`)

## AI Route Optimization (new)

The navigation layer now goes beyond a single A* route. For the same origin and destination it can generate several distinct loopless candidate routes using **Yen's K-shortest-path algorithm**, then select among those candidates using an explicit objective:

- **Shortest distance** — minimize route length.
- **Fastest estimated time** — minimize simulated travel time.
- **Balanced** — combine normalized distance and estimated time.

The dashboard also benchmarks **BFS, DFS, UCS, Greedy Best-First, and A*** on the same query and reports:

- route distance
- estimated travel time
- nodes expanded
- measured execution time
- theoretical time-complexity class
- whether the algorithm guarantees distance optimality on this weighted graph

### Multi-Vehicle Optimization

A project-scale **Genetic Algorithm** is included for assigning multiple healthcare destinations across multiple vehicles. The fitness function balances total travel distance with route imbalance, and the UI reports each vehicle's route, total distance, maximum vehicle distance, and balance gap.

> **Scope:** the graph and travel speeds are synthetic for the academic prototype. They are not live Google Maps traffic or road data. The architecture is intentionally separated so a real GIS/routing provider can replace the synthetic graph later.

## Project layout

```
healthcare-ai-agent/
├── app/            # frontend (Streamlit) + backend (FastAPI, later)
├── agents/         # supervisor + specialized agents
├── ai/
│   ├── search/     # bfs, dfs, ucs, greedy, astar
│   ├── csp/        # appointment scheduling CSP
│   ├── knowledge/  # facts, rules, forward/backward chaining
│   ├── expert_system/
│   ├── ml/         # decision tree, k-means
│   └── vision/     # OCR / document understanding
├── data/           # facilities.csv, services.csv, appointments.csv, feedback.csv
├── evaluation/      # search/ML/agent metrics
├── docs/           # design docs (this is where Step 1 lives)
└── tests/
```

## Run (once modules exist)

```bash
pip install -r requirements.txt
streamlit run app/frontend/app.py
```
