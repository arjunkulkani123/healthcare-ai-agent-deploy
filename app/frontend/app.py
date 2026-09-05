"""
Healthcare Experience Agent -- Streamlit Dashboard (Multi-Tab Showcase)
============================================================================

Tab 1 (Assistant)         -- the end-to-end agent, as before.
Tab 2 (Search Algorithms) -- live BFS/DFS/UCS/Greedy/A* comparison with a
                              visual graph plot highlighting the A* path.
Tab 3 (CSP Scheduling)    -- naive vs smart backtracking comparison plus
                              ranked appointment options.
Tab 4 (ML Insights)       -- Decision Tree accuracy/confusion matrix and
                              K-Means cluster visualization, computed live.
Tab 5 (Knowledge Base)    -- forward chaining trace + a goal-driven
                              backward chaining query.

Run with:
    streamlit run app/frontend/app.py
"""

import sys
import os

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "agents"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "vision"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "search"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "csp"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "ml"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "knowledge"))

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from supervisor_agent import handle_request

st.set_page_config(page_title="Healthcare Experience Agent", page_icon="\u2695", layout="wide")

st.markdown(
    """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            color-scheme: light !important;
        }
        :root {
            --teal-900: #0b3d3a;
            --teal-700: #0f6b63;
            --teal-500: #14919b;
            --cream: #f7f5f0;
            --ink: #1c2521;
        }
        .stApp { background-color: var(--cream); }
        /* Base default: color everything inside the app dark-ink by
           default (no !important here, so specific rules below like
           badges/buttons/headers can still override it). This is a
           broad safety net covering st.text(), st.caption(), st.code(),
           st.metric(), st.warning(), plain markdown, and anything else
           we haven't individually targeted -- fixing them all in one
           shot instead of chasing each widget type separately. */
        [data-testid="stAppViewContainer"] * {
            color: var(--ink);
        }
        h1, h2, h3, h4, h5, h6 { color: var(--teal-900) !important; font-family: 'Georgia', serif; }
        .subtitle { color: #5a6b66; font-size: 1.05rem; margin-top: -0.6rem; margin-bottom: 1.5rem; }
        .trace-step {
            padding: 0.5rem 0.8rem;
            border-left: 3px solid var(--teal-500);
            background: #ffffff;
            margin-bottom: 0.4rem;
            border-radius: 0 6px 6px 0;
            font-size: 0.92rem;
            color: var(--ink);
        }
        .result-card {
            background: #ffffff;
            border: 1px solid #e2e2dc;
            border-radius: 10px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
            color: var(--ink);
        }
        .result-card p, .result-card h4, .result-card li {
            color: var(--ink) !important;
        }
        .badge {
            display: inline-block;
            padding: 0.15rem 0.7rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .badge-high { background: #fde2e1; color: #9c2b26; }
        .badge-medium { background: #fdf0d0; color: #8a5e10; }
        .badge-routine { background: #dcf0e6; color: #1f6b46; }
        .scope-note {
            font-size: 0.85rem;
            color: #78877f;
            border-top: 1px solid #e2e2dc;
            padding-top: 0.7rem;
            margin-top: 0.9rem;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stMarkdownContainer"] p {
            color: var(--ink) !important;
        }
        [data-testid="stButton"] p,
        [data-testid="stButton"] button p {
            color: #ffffff !important;
        }
        [data-testid="stExpander"] {
            background: #ffffff;
            border-radius: 10px;
        }
        [data-testid="stExpanderDetails"] {
            background: #ffffff;
            color: var(--ink) !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpanderHeader"],
        [data-testid="stExpander"] details {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpanderHeader"]:hover {
            background: #fafaf7 !important;
        }
        [data-testid="stExpander"] summary svg,
        [data-testid="stExpanderHeader"] svg {
            fill: var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #fafaf7 !important;
            border: 1px dashed #cfcfc7 !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] svg {
            fill: var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: #ececE6 !important;
            border: none !important;
            color: var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #e0ddd4 !important;
        }
        [data-testid="stFileUploaderDropzone"] button svg,
        [data-testid="stFileUploaderDropzone"] button p {
            fill: var(--ink) !important;
            color: var(--ink) !important;
        }
        [data-testid="stTabs"] button p {
            color: var(--ink) !important;
            font-weight: 600;
        }
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        [data-testid="stTable"] table {
            color: var(--ink) !important;
            background: #ffffff !important;
        }
        [data-testid="stTable"] th {
            color: var(--teal-900) !important;
            background: #f0efe9 !important;
        }
        [data-testid="stTable"] td {
            color: var(--ink) !important;
        }
        /* Placed LAST and deliberately: an earlier button-color rule
           had the same CSS specificity as the markdown-container ink
           rule above, and since both use !important, the LATER one in
           the file wins regardless of intent -- that's why button text
           kept losing to the dark ink color. Putting this fix last
           guarantees it wins the tie. */
        button, [data-testid="stButton"], [data-testid="stButton"] button {
            color-scheme: light !important;
        }
        [data-testid="stButton"] button,
        [data-testid="stButton"] button * {
            color: #ffffff !important;
        }
        /* st.json() and st.code() use JS libraries (react-json-view,
           Pygments-style highlighting) that set their own inline color
           per token AND their own dark background by default. Forcing
           only the text color (previous attempt) left dark text on a
           still-dark background -- invisible. Fixing both together. */
        [data-testid="stJson"],
        [data-testid="stCode"],
        pre {
            background: #ffffff !important;
            border: 1px solid #e2e2dc !important;
        }
        [data-testid="stJson"] *,
        [data-testid="stCode"] *,
        pre, code, pre *, code * {
            color: var(--ink) !important;
            background: transparent !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# Healthcare Experience Agent")
st.markdown(
    '<p class="subtitle">An agentic AI system combining search, constraint '
    "satisfaction, expert-system reasoning, and machine learning to "
    "navigate healthcare services.</p>",
    unsafe_allow_html=True,
)

tab_assistant, tab_search, tab_csp, tab_ml, tab_kb = st.tabs(
    ["\U0001F4AC Assistant", "\U0001F5FA Search Algorithms", "\U0001F4C5 CSP Scheduling",
     "\U0001F4CA ML Insights", "\U0001F9E0 Knowledge Base"]
)

# =======================================================================
# TAB 1: ASSISTANT
# =======================================================================
with tab_assistant:
    EXAMPLES = {
        "Urgent senior, budget-conscious": (
            "My mother has had a high fever since yesterday and I need to "
            "find a government hospital nearby. She is 62 and we don't have "
            "much money."
        ),
        "Routine vaccination": (
            "I'm 28 years old and need a vaccination appointment tomorrow "
            "morning, preferably at a government facility nearby."
        ),
        "Chronic condition, mobility-impaired": (
            "My father is 70 and has a chronic condition, he needs a "
            "specialist consultation. He has trouble walking so we need "
            "somewhere within 8 km."
        ),
    }

    def _use_example(text):
        st.session_state["user_text"] = text

    col_input, col_examples = st.columns([3, 1])
    with col_input:
        user_text = st.text_area(
            "How can I help you?",
            height=100,
            placeholder="e.g. My mother has had a high fever since yesterday and I need to find a government hospital nearby...",
            key="user_text",
        )
    with col_examples:
        st.markdown("**Try an example**")
        for label, text in EXAMPLES.items():
            st.button(label, use_container_width=True, on_click=_use_example, args=(text,), key="ex_" + label)

    run_clicked = st.button("Ask Agent", type="primary")

    if run_clicked and st.session_state.get("user_text", "").strip():
        with st.spinner("Reasoning through the request..."):
            result = handle_request(st.session_state["user_text"])

        left, right = st.columns([1, 1.4])

        with left:
            st.markdown("### Agent Trace")
            for step in result["trace"]:
                st.markdown('<div class="trace-step">' + step + '</div>', unsafe_allow_html=True)

        with right:
            st.markdown("### Recommendation")
            urgency = result["expert_result"]["urgency_level"] or "routine"
            badge_class = {"high": "badge-high", "medium": "badge-medium"}.get(urgency, "badge-routine")

            st.markdown(
                '<div class="result-card">'
                '<span class="badge ' + badge_class + '">' + urgency.upper() + ' URGENCY</span>'
                '<span class="badge badge-routine">' + result['expert_result']['recommended_service'].replace('_', ' ') + '</span>'
                '<p style="margin-top:0.8rem; white-space: pre-line;">' + result['expert_result']['explanation'] + '</p>'
                '</div>',
                unsafe_allow_html=True,
            )

            if result["appointment"]:
                appt = result["appointment"]
                st.markdown(
                    '<div class="result-card">'
                    '<h4 style="margin-top:0;">Appointment</h4>'
                    '<p><b>' + appt['doctor'] + '</b> at <b>' + appt['hospital'] + '</b><br>'
                    + appt['date'] + ' at ' + appt['time'] + '</p>'
                    '<p style="white-space: pre-line; font-size: 0.9rem;">' + result['appointment_explanation'] + '</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                if result["route"] and result["route"]["path"]:
                    route = result["route"]
                    path_str = " &rarr; ".join(route["path"])
                    st.markdown(
                        '<div class="result-card">'
                        '<h4 style="margin-top:0;">Suggested Route</h4>'
                        '<p>' + path_str + '</p>'
                        '<p style="color:#5a6b66; font-size:0.85rem;">'
                        'Approx. ' + str(route['cost']) + ' km &middot; A* expanded ' + str(route['nodes_expanded']) + ' nodes'
                        '</p></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(
                    "No appointment slot satisfies all constraints. Try relaxing "
                    "the time window or distance limit."
                )

            st.markdown(
                '<p class="scope-note">This is a service-navigation recommendation, '
                "not a medical diagnosis. Please consult a qualified clinician for "
                "any medical concerns.</p>",
                unsafe_allow_html=True,
            )
    elif run_clicked:
        st.warning("Please describe what you need first.")

    st.markdown("---")
    with st.expander("\U0001F4CE Upload a document (appointment slip, registration form)"):
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                from document_processor import process_document
                with st.spinner("Reading document..."):
                    doc_result = process_document(tmp_path)
                st.image(uploaded_file, width=300)
                st.markdown("**Agent response:**")
                st.write(doc_result["response"])
                with st.expander("Raw extracted fields"):
                    st.json(doc_result["fields"])
            except Exception as e:
                st.error("Couldn't process this document: " + str(e))


# =======================================================================
# TAB 2: SEARCH ALGORITHMS
# =======================================================================
with tab_search:
    st.markdown("### Compare BFS, DFS, UCS, Greedy, and A* on the same query")
    st.caption(
        "All five algorithms search the same healthcare navigation graph. "
        "Node coordinates are real (x, y) positions, so the A*/Greedy "
        "heuristic (straight-line distance) is provably admissible."
    )

    from graph import build_graph, NODE_COORDS
    from bfs import bfs
    from dfs import dfs
    from ucs import ucs
    from greedy import greedy
    from astar import astar

    node_names = sorted(NODE_COORDS.keys())
    col1, col2 = st.columns(2)
    with col1:
        start_node = st.selectbox("Start", node_names, index=node_names.index("Home"))
    with col2:
        goal_node = st.selectbox("Goal", node_names, index=node_names.index("Laboratory_A"))

    if st.button("Run comparison", type="primary"):
        graph = build_graph()
        algorithms = {"BFS": bfs, "DFS": dfs, "UCS": ucs, "Greedy": greedy, "A*": astar}
        results = {}
        for name, fn in algorithms.items():
            results[name] = fn(graph, start_node, goal_node)

        st.markdown("#### Results")
        table_data = []
        for name, r in results.items():
            table_data.append({
                "Algorithm": name,
                "Nodes Expanded": r["nodes_expanded"],
                "Path Length": r["path_length_edges"],
                "Cost (km)": r["cost"],
                "Time (ms)": r["time_taken_ms"],
            })
        st.table(table_data)

        fig, ax = plt.subplots(figsize=(8, 6))
        for node, coords in NODE_COORDS.items():
            x, y = coords
            ax.scatter(x, y, s=40, color="#cfcfc7", zorder=2)
            ax.annotate(node, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#1c2521")
        for node, neighbors in graph.items():
            x1, y1 = NODE_COORDS[node]
            for neighbor, _w in neighbors:
                x2, y2 = NODE_COORDS[neighbor]
                ax.plot([x1, x2], [y1, y2], color="#e2e2dc", linewidth=1, zorder=1)

        astar_path = results["A*"]["path"]
        if astar_path:
            path_x = [NODE_COORDS[n][0] for n in astar_path]
            path_y = [NODE_COORDS[n][1] for n in astar_path]
            ax.plot(path_x, path_y, color="#14919b", linewidth=3, zorder=3, label="A* path")
            ax.scatter(path_x, path_y, s=60, color="#0b3d3a", zorder=4)

        ax.set_title("A* path: " + start_node + " to " + goal_node)
        ax.legend()
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        st.pyplot(fig)

        st.markdown("#### Paths found")
        for name, r in results.items():
            path_str = " \u2192 ".join(r["path"]) if r["path"] else "No path found"
            st.markdown("**" + name + "**: " + path_str)

        st.info(
            "BFS minimizes hop count (not distance). UCS and A* both "
            "guarantee the true lowest-cost path -- A* typically reaches "
            "it while expanding fewer nodes because its heuristic focuses "
            "the search toward the goal. Greedy is fast but can return a "
            "suboptimal path since it ignores accumulated cost."
        )


# =======================================================================
# TAB 3: CSP SCHEDULING
# =======================================================================
with tab_csp:
    st.markdown("### Appointment scheduling as a Constraint Satisfaction Problem")
    st.caption(
        "Compares naive backtracking against smart backtracking (MRV "
        "variable ordering + forward checking) on the same request."
    )

    from domain import PatientRequest, ExistingBooking, build_domains
    from constraints import build_constraints
    from backtracking import naive_backtracking, smart_backtracking
    from ranking import rank_solutions, explain as explain_appointment

    col1, col2, col3 = st.columns(3)
    with col1:
        service = st.selectbox("Service needed", ["OPD", "Vaccination", "Emergency", "Diagnostic"])
    with col2:
        time_pref = st.selectbox("Preferred time", ["morning", "afternoon", "evening"])
    with col3:
        max_dist = st.slider("Max distance (km)", 1, 20, 10)

    if st.button("Solve", type="primary"):
        request = PatientRequest(
            service=service, preferred_time_of_day=time_pref,
            max_distance_km=float(max_dist), preferred_dates=["Mon", "Tue", "Wed"],
        )
        existing_bookings = [
            ExistingBooking(doctor="Dr_Rao", date="Mon", time="10:00"),
            ExistingBooking(doctor="Dr_Iyer", date="Mon", time="10:00"),
        ]
        domains = build_domains(request)
        constraints = build_constraints(request, existing_bookings)
        variables = ["doctor", "hospital", "date", "time"]

        naive_result = naive_backtracking(variables, domains, constraints, limit=10)
        smart_result = smart_backtracking(variables, domains, constraints, limit=10)

        st.markdown("#### Naive vs Smart Backtracking")
        st.table(
            [
                {"Method": "Naive Backtracking", "Nodes Expanded": naive_result["nodes_expanded"], "Time (ms)": naive_result["time_taken_ms"]},
                {"Method": "Smart (MRV + Forward Checking)", "Nodes Expanded": smart_result["nodes_expanded"], "Time (ms)": smart_result["time_taken_ms"]},
            ],
        )

        solutions = smart_result["solutions"]
        if solutions:
            ranked = rank_solutions(solutions, request)
            st.markdown("#### " + str(len(ranked)) + " feasible appointment(s) found")
            for i, sol in enumerate(ranked[:5], start=1):
                st.markdown(str(i) + ". **" + sol['doctor'] + "** @ " + sol['hospital'] + " -- " + sol['date'] + " " + sol['time'])

            st.markdown("#### Top recommendation")
            st.markdown(
                '<div class="result-card"><p style="white-space:pre-line;">' + explain_appointment(ranked[0], request) + '</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No feasible appointment found with these constraints.")


# =======================================================================
# TAB 4: ML INSIGHTS
# =======================================================================
with tab_ml:
    st.markdown("### Decision Tree -- Service Category Classification")

    from decision_tree import train_and_evaluate
    from kmeans import run_kmeans, profile_clusters, label_cluster

    if st.button("Train Decision Tree", type="primary"):
        with st.spinner("Training..."):
            dt_results = train_and_evaluate()

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", "{:.1f}%".format(dt_results['accuracy'] * 100))
        col2.metric("Train / Test size", str(dt_results['n_train']) + " / " + str(dt_results['n_test']))

        st.text("Classification report:")
        st.code(dt_results["classification_report"])

        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = dt_results["confusion_matrix"]
        labels = dt_results["labels"]
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=9)
        fig.colorbar(im)
        st.pyplot(fig)

        st.markdown("#### Feature Importances")
        importances = dt_results["feature_importances"]
        st.bar_chart(importances)

    st.markdown("---")
    st.markdown("### K-Means -- Patient Experience Segmentation")

    if st.button("Run K-Means Clustering", type="primary"):
        with st.spinner("Clustering..."):
            km_result = run_kmeans()

        st.metric("Silhouette Score", "{:.3f}".format(km_result['silhouette_score']))

        profiles = profile_clusters(km_result["rows"], km_result["k"])
        table_data = []
        for cid, profile in profiles.items():
            table_data.append({
                "Cluster": cid,
                "Label": label_cluster(profile),
                "Size": profile["count"],
                "Avg Wait (min)": profile["waiting_time_minutes"],
                "Avg Satisfaction": profile["satisfaction_score"],
            })
        st.table(table_data)

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#14919b", "#9c2b26", "#8a5e10", "#1f6b46"]
        for cid in profiles:
            members = [r for r in km_result["rows"] if r["cluster"] == cid]
            xs = [m["waiting_time_minutes"] for m in members]
            ys = [m["satisfaction_score"] for m in members]
            ax.scatter(xs, ys, s=15, alpha=0.6, color=colors[cid % len(colors)], label="Cluster " + str(cid))
        ax.set_xlabel("Waiting time (minutes)")
        ax.set_ylabel("Satisfaction score")
        ax.legend()
        st.pyplot(fig)


# =======================================================================
# TAB 5: KNOWLEDGE BASE
# =======================================================================
with tab_kb:
    st.markdown("### Forward Chaining -- Data-Driven Reasoning")

    from forward_chaining import forward_chain
    from backward_chaining import backward_chain
    from facts import EXAMPLE_CASE_URGENT_SENIOR, EXAMPLE_CASE_ROUTINE_VACCINATION, EXAMPLE_CASE_CHRONIC_FOLLOWUP

    scenario_options = {
        "Urgent senior, high fever": EXAMPLE_CASE_URGENT_SENIOR,
        "Routine vaccination": EXAMPLE_CASE_ROUTINE_VACCINATION,
        "Chronic condition follow-up": EXAMPLE_CASE_CHRONIC_FOLLOWUP,
    }
    chosen = st.selectbox("Choose a scenario", list(scenario_options.keys()))

    if st.button("Run Forward Chaining", type="primary"):
        facts = scenario_options[chosen]
        result = forward_chain(facts)

        st.markdown("#### Facts")
        st.json(result["facts"])

        st.markdown("#### Rules fired, in order")
        for rule_name, key, value in result["trace"]:
            st.markdown('<div class="trace-step">' + rule_name + ": derived <b>" + key + " = " + str(value) + "</b></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Backward Chaining -- Goal-Driven Reasoning")
    st.caption("Given partial facts, work backwards to see what's provable and what's still missing.")

    goal_service = st.selectbox("Goal: can we conclude the service is...", ["Emergency", "Urgent_OPD", "Vaccination", "General_OPD"])
    known_age = st.number_input("Known: patient age", min_value=0, max_value=120, value=62)
    known_fever = st.checkbox("Known: has fever", value=True)
    known_gov_pref = st.checkbox("Known: prefers government facility", value=True)

    if st.button("Run Backward Chaining", type="primary"):
        partial_facts = {"age": known_age, "has_fever": known_fever, "government_preference": known_gov_pref}
        result = backward_chain(("recommended_service", goal_service), partial_facts)

        st.metric("Proved?", "Yes" if result["proved"] else "No")
        if result["missing_facts"]:
            st.warning("Facts still needed to confirm this: " + ", ".join(result['missing_facts']))
            st.caption("In the live agent, these become the next follow-up question asked to the user.")

        st.markdown("#### Reasoning trace")
        trace_lines = []
        for step in result["trace"]:
            indent = "    " * step["depth"]
            icon = "\u2713" if step["result"] else "\u2717"
            trace_lines.append(indent + icon + " " + str(step["goal"]) + " -- " + step["reason"])
        st.code("\n".join(trace_lines), language=None)
