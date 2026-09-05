# 🏥 Healthcare Experience Agent

**An agentic AI system that helps citizens navigate public healthcare services** — classifying urgency, scheduling real appointment slots, planning routes, and explaining every decision along the way.

🔗 **[Live Demo](https://healthcare-ai-agent-deploy-kjfc6f8u2hmjx5npdudtco.streamlit.app/)** &nbsp;|&nbsp; Built with Python, Streamlit, and scikit-learn

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-ff4b4b?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this is

Public healthcare systems are hard to navigate: long waits, fragmented information, unclear eligibility, and difficulty finding the right facility. This project is an **agentic AI system** that combines *classical AI* (search, constraint satisfaction, rule-based reasoning) with *machine learning* to solve that navigation problem end-to-end — and explains every step of its own reasoning.

**Scope boundary:** this system recommends and schedules healthcare *services* — it does not diagnose disease or recommend treatment. Symptom text is used only to classify urgency and service category (e.g. Emergency vs. OPD vs. Vaccination), never to output a medical conclusion.

> Example: *"My mother has had a high fever since yesterday and I need to find a government hospital nearby. She is 62 and we don't have much money."*
> → classified as **Emergency**, routed to a **public facility**, an **appointment slot found and ranked**, and a **route computed** — with a plain-English explanation for each step.

## Live features (5 tabs)

| Tab | What it demonstrates |
|---|---|
| 💬 **Assistant** | The full end-to-end agent: free-text request → classification → scheduling → routing → explanation, plus a document-upload (OCR) flow |
| 🗺️ **Search Algorithms** | Live BFS / DFS / UCS / Greedy / A\* comparison on a real coordinate-based navigation graph, with a visual path plot |
| 📅 **CSP Scheduling** | Naive vs. smart (MRV + forward checking) backtracking for appointment scheduling, with live constraint tuning |
| 📊 **ML Insights** | A Decision Tree trained live (accuracy, confusion matrix, feature importances) and K-Means patient-experience clustering |
| 🧠 **Knowledge Base** | Forward chaining (data-driven) and backward chaining (goal-driven) over a rule-based expert system |

## AI techniques used

This project was built to demonstrate a full undergraduate AI syllabus in one coherent system, not as isolated exercises:

- **Search:** BFS, DFS, Uniform Cost Search, Greedy Best-First, A\* — with a provably admissible heuristic (straight-line distance, verified via the triangle inequality)
- **Constraint Satisfaction:** backtracking with MRV variable ordering + forward checking, benchmarked against naive backtracking
- **Knowledge Representation & Reasoning:** propositional rule base with forward chaining (data-driven) and backward chaining (goal-driven) inference
- **Expert Systems:** rule-based classification with a self-explanation layer
- **Machine Learning:** Decision Tree classification (with realistic injected label noise, not a suspicious 100% accuracy) and K-Means unsupervised clustering, both properly evaluated (train/test split, confusion matrix, silhouette score)
- **Computer Vision:** OCR-based document understanding (Tesseract) for appointment slips/registration forms, with image preprocessing and field validation
- **Agentic Architecture:** a Supervisor Agent that perceives, plans, uses tools, acts, and explains — with an LLM-ready natural-language-understanding layer (Claude API integration, with automatic fallback to a rule-based parser)

## Architecture

```
USER REQUEST (free text)
        │
        ▼
  Supervisor Agent
        │
   ┌────┼─────────────┬──────────────┐
   ▼    ▼              ▼              ▼
 NLU   Expert System   CSP Solver    A* Search
       (forward         (MRV +        (admissible
        chaining)     forward check)   heuristic)
   │    │              │              │
   └────┴──────┬───────┴──────────────┘
                ▼
      Recommendation + Appointment
        + Route + Explanation
```

The LLM (where enabled) handles only language understanding; every decision — urgency classification, scheduling, routing — is made by deterministic, explainable, classical AI. This split is deliberate: it keeps the system auditable.

## Tech stack

- **Frontend:** Streamlit
- **ML:** scikit-learn (Decision Tree, K-Means)
- **OCR:** Tesseract + Pillow
- **LLM (optional):** Anthropic Claude API, with graceful fallback
- **Real-world data (optional):** Google Places API integration for real facility lookup
- **Deployment:** Streamlit Community Cloud + GitHub

## Run it locally

```bash
git clone https://github.com/arjunkulkani123/healthcare-ai-agent-deploy.git
cd healthcare-ai-agent-deploy
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
streamlit run app/frontend/app.py
```

Optional: add `ANTHROPIC_API_KEY` (for LLM-based understanding) or `GOOGLE_PLACES_API_KEY` (for real facility data) as environment variables or in `.streamlit/secrets.toml` — the app works fully without them, using rule-based fallbacks.

## Project structure

```
healthcare-ai-agent-deploy/
├── agents/              # Supervisor agent + NLU (regex and LLM-based)
├── ai/
│   ├── search/          # BFS, DFS, UCS, Greedy, A*
│   ├── csp/              # Constraint satisfaction (backtracking, ranking)
│   ├── knowledge/        # Rules, forward/backward chaining
│   ├── expert_system/    # Rule-based classification + explanation
│   ├── ml/                # Decision Tree, K-Means
│   ├── vision/            # OCR document processing
│   └── data/              # Real facility data integration (Google Places)
├── app/frontend/         # Streamlit dashboard (5-tab showcase)
├── evaluation/           # Automated evaluation across every module
└── docs/                 # Problem statement, PEAS, use cases
```

## Evaluation highlights

Real numbers, generated by `evaluation/run_all_evaluations.py` — not hand-picked:

- **A\*** matches the optimal path cost of Uniform Cost Search while expanding fewer-or-equal nodes
- **Smart backtracking** (MRV + forward checking) cuts CSP search nodes by ~59% vs. naive backtracking
- **Decision Tree:** 92% accuracy (deliberately not 100% — 5% label noise injected to avoid overfitting)
- **K-Means:** 0.30 silhouette score, correctly rediscovering 4 real-world patient-experience segments
- **Supervisor Agent:** 100% task completion across varied test scenarios, sub-2ms average response time

## License

MIT — feel free to fork, learn from, or build on this project.

---

*Built as an academic project demonstrating classical AI (search, CSP, knowledge representation, expert systems) and machine learning (supervised + unsupervised), architected as a genuine multi-tool agentic system rather than a single-model chatbot.*
