# Step 1 — Problem Statement, Users, Use Cases, PEAS

## 1. Problem Statement

Public healthcare systems often involve complex service navigation, long
waiting times, fragmented information, eligibility constraints, and
difficulty identifying the appropriate healthcare facility. This project
proposes an **agentic AI system** — the Healthcare Experience Agent — that
combines knowledge-based reasoning, classical search, constraint
satisfaction, machine learning, and conversational AI to optimize a
citizen's end-to-end healthcare **service-navigation experience**.

**Scope boundary (important for viva defensibility):** the system
recommends, schedules, and explains healthcare *services and facilities*.
It does **not** diagnose disease or recommend treatment. Symptom text is
used only to classify **urgency** and **service category** (e.g.
"Emergency" vs "OPD" vs "Vaccination"), never to output a medical
diagnosis.

## 2. Users / Personas

| Persona | Description | Primary need |
|---|---|---|
| Citizen / Patient | General public seeking a healthcare service | Find the right facility fast |
| Caregiver | Booking/navigating on behalf of a dependent (e.g. elderly parent) | Simplicity, low friction, trust |
| Frequent visitor | Chronic condition, repeat visits | Continuity, fast repeat booking |
| Digitally underserved user | Limited literacy / connectivity | Simple language, minimal steps |
| Administrator (secondary) | Healthcare facility staff | Feedback/analytics dashboard (optional stretch) |

## 3. Use Cases (5–7)

1. **Urgent symptom triage** — "My mother has had a high fever since
   yesterday, she is 62, we don't have much money." → classify urgency,
   recommend nearest appropriate *public* facility, explain reasoning.
2. **Routine OPD navigation** — "I need a general check-up this week." →
   find OPD slot, minimize travel + wait, propose appointment.
3. **Constrained appointment scheduling** — Patient wants a morning slot,
   doctor available 10 AM–2 PM only, hospital open Mon–Fri, patient can't
   travel >10 km → CSP solves feasible slot.
4. **Vaccination eligibility & location** — "Vaccination center near me
   tomorrow morning" → eligibility rule check + nearest-available search.
5. **Document-assisted booking** — User uploads a photo of a referral
   slip / registration form → OCR extracts date, facility, service →
   agent pre-fills booking flow.
6. **Post-visit feedback & experience optimization** — After a visit, user
   rates wait time/friction → feeding the clustering + optimization loop.
7. **Cost/public-preference constrained search** — "We don't have much
   money" → filter to public/free facilities before ranking by
   distance/urgency.

## 4. PEAS Specification

| Element | Details |
|---|---|
| **Performance measure** | Reduced waiting time; improved service-match accuracy; user satisfaction score; reduced navigation steps; information accuracy; constraint compliance (budget, distance, eligibility) |
| **Environment** | Citizens, hospitals/clinics, healthcare services catalog, appointment slots, transportation/distance data, feedback history |
| **Actuators** | Recommend facility; generate route/path; prioritize among options; issue instructions (documents, timing); ask clarifying questions; generate follow-up/feedback prompts |
| **Sensors** | User's natural-language text; uploaded documents/images; location; stated preferences (cost, public/private); symptom/urgency cues; facility metadata; historical feedback/wait-time data |

**Environment properties:** partially observable (we don't know real-time
hospital queue length exactly — estimated), stochastic (wait times vary),
sequential (a session has multi-turn state), dynamic (slot availability
changes), discrete (finite facilities/services/slots), multi-agent in the
weak sense (many citizens compete for the same slots/capacity).

## 5. High-Level Architecture (recap)

```
USER → Supervisor Agent → {Knowledge Agent, Search/CSP Agent, ML Agent}
     → Expert System (rules + explanation) → Recommendation
     → Action → Feedback → K-Means experience analysis → Optimizer
     → improves future recommendations
```

The LLM is restricted to: language understanding, planning/tool
selection, and explanation generation. All search, constraints, rules,
classification, and clustering are classical/deterministic AI — this is
the architectural point to lead with in the viva.

## 6. Next Step

Step 2 → Design the healthcare navigation graph and implement
BFS / DFS / UCS / Greedy / A* over it (`ai/search/`).
