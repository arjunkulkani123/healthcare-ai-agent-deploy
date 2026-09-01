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
