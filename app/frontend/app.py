"""
Healthcare Experience Agent -- Streamlit Dashboard (Multi-Tab Showcase)

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
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "optimization", "routing"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "optimization", "multi_vehicle"))

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
        h1, h2, h3 { color: var(--teal-900) !important; font-family: 'Georgia', serif; }
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
               /* Force readable text color on Streamlit's own widget labels,
           markdown text, and expander headers -- these don't go through
           our custom .result-card styling, so without this they can
           inherit a very light default color that's hard to read
           against our light background. Deliberately excludes .stButton
           so button text keeps its own (white-on-dark) styling. */
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
               [data-testid="stButton"] p,
        [data-testid="stButton"] button p {
            color: #ffffff !important;
        }
        /* The expander body and file-upload dropzone use Streamlit's own
           default styling (often a dark background), which our custom
           light theme never touched -- fix them explicitly. */
        [data-testid="stExpander"] {
            background: #ffffff;
            border-radius: 10px;
        }
        [data-testid="stExpanderDetails"] {
            background: #ffffff;
            color: var(--ink) !important;
        }
        /* The expander HEADER (the clickable bar itself) ships with its
           own dark background by default, which none of the rules above
           touch -- that's the solid black bar in the screenshot. Override
           it explicitly so it matches the rest of the light theme. Covers
           both the current Streamlit testid and the underlying <summary>
           element for older/newer versions. */
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
        /* The "Upload" button inside the dropzone defaults to a solid
           black fill with white text/icon -- switch it to a light gray
           pill matching the rest of the light theme. */
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
                if result["route"] and result["route"].get("selected_route"):
                    route = result["route"]
                    selected_route = route["selected_route"]
                    path_str = " &rarr; ".join(selected_route["path"])
                    st.markdown(
                        '<div class="result-card">'
                        '<h4 style="margin-top:0;">AI-Optimized Route</h4>'
                        '<p><b>Route ' + str(selected_route["rank"]) + '</b>: ' + path_str + '</p>'
                        '<p style="color:#5a6b66; font-size:0.85rem;">' +
                        str(selected_route["distance_km"]) + ' km &middot; about ' + str(selected_route["estimated_time_min"]) + ' min &middot; ' +
                        str(len(route["routes"])) + ' alternatives compared</p>'
                        '</div>',
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
# TAB 2: AI ROUTE OPTIMIZATION
# =======================================================================
with tab_search:
    st.markdown("### AI Route Optimization — multiple routes, algorithm comparison, and multi-vehicle planning")
    st.caption(
        "The graph is a synthetic healthcare road network. The system generates several distinct routes, "
        "benchmarks five search algorithms, reports computational cost, and can optimize a small multi-vehicle scenario."
    )

    from graph import build_graph, NODE_COORDS
    from route_optimizer import optimize_routes, benchmark_algorithms, k_shortest_paths
    from ga_vrp import optimize_multi_vehicle

    node_names = sorted(NODE_COORDS.keys())
    col1, col2, col3 = st.columns(3)
    with col1:
        start_node = st.selectbox("Start", node_names, index=node_names.index("Home"), key="route_start")
    with col2:
        goal_node = st.selectbox("Destination", node_names, index=node_names.index("Laboratory_A"), key="route_goal")
    with col3:
        objective_label = st.selectbox("Optimization objective", ["Balanced", "Shortest distance", "Fastest estimated time"])

    objective_map = {"Balanced": "balanced", "Shortest distance": "shortest", "Fastest estimated time": "fastest"}
    k_routes = st.slider("Alternative routes to generate", 2, 5, 3)

    if st.button("Optimize Routes", type="primary"):
        graph = build_graph()
        optimization = optimize_routes(
            graph, start_node, goal_node, k=k_routes, objective=objective_map[objective_label]
        )
        st.session_state["route_optimization"] = optimization
        st.session_state["route_graph"] = graph
        st.session_state["route_query"] = (start_node, goal_node)

    optimization = st.session_state.get("route_optimization")
    graph = st.session_state.get("route_graph")

    if optimization and graph:
        selected = optimization.get("selected_route")
        st.markdown("#### Candidate routes")
        route_rows = []
        for route in optimization["routes"]:
            route_rows.append({
                "Route": f"Route {route['rank']}",
                "Distance (km)": route["distance_km"],
                "Estimated time (min)": route["estimated_time_min"],
                "Hops": route["hops"],
                "Objective score": route["objective_score"],
                "Selected": "✓" if selected and route["rank"] == selected["rank"] else "",
            })
        st.dataframe(route_rows, use_container_width=True, hide_index=True)

        if selected:
            st.success(
                f"Selected Route {selected['rank']} — {selected['distance_km']} km, "
                f"about {selected['estimated_time_min']} min — objective: {optimization['objective']}."
            )
            st.code(" → ".join(selected["path"]), language="text")

        st.markdown("#### Search algorithm comparison")
        benchmark_rows = []
        for row in optimization["benchmark"]:
            benchmark_rows.append({
                "Algorithm": row["algorithm"],
                "Distance (km)": row["distance_km"],
                "Est. time (min)": row["estimated_time_min"],
                "Nodes expanded": row["nodes_expanded"],
                "Execution (ms)": row["execution_time_ms"],
                "Time complexity": row["complexity"],
                "Distance-optimal": "Yes" if row["optimal_for_distance"] else "No",
            })
        st.dataframe(benchmark_rows, use_container_width=True, hide_index=True)

        st.markdown("#### What the comparison means")
        st.info(
            "BFS and DFS are O(V+E) graph traversals but do not optimize weighted distance. "
            "UCS and A* are distance-optimal on this non-negative weighted graph. A* uses the admissible "
            "straight-line heuristic to focus the search. Greedy Best-First uses only the heuristic, so it can "
            "be faster but is not guaranteed to return the shortest weighted route. The measured execution time "
            "is hardware/runtime dependent; the theoretical complexity is the defensible comparison for your viva."
        )

        # Synthetic map view with every candidate route.
        st.markdown("#### Route map (synthetic graph)")
        fig, ax = plt.subplots(figsize=(9, 6))
        for node, coords in NODE_COORDS.items():
            x, y = coords
            ax.scatter(x, y, s=45, color="#cfcfc7", zorder=2)
            ax.annotate(node, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#1c2521")
        for node, neighbors in graph.items():
            x1, y1 = NODE_COORDS[node]
            for neighbor, _w in neighbors:
                x2, y2 = NODE_COORDS[neighbor]
                ax.plot([x1, x2], [y1, y2], color="#e2e2dc", linewidth=1, zorder=1)
        for idx, route in enumerate(optimization["routes"]):
            xs = [NODE_COORDS[n][0] for n in route["path"]]
            ys = [NODE_COORDS[n][1] for n in route["path"]]
            linewidth = 3.5 if selected and route["rank"] == selected["rank"] else 1.8
            alpha = 0.9 if selected and route["rank"] == selected["rank"] else 0.45
            ax.plot(xs, ys, linewidth=linewidth, alpha=alpha, zorder=3, label=f"Route {route['rank']}")
        ax.set_title(f"Candidate routes: {start_node} → {goal_node}")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("#### Multi-Vehicle Optimization (Genetic Algorithm)")
        st.caption("Assign several healthcare destinations across vehicles while minimizing total travel and route imbalance.")
        vehicle_candidates = [n for n in ["Govt_Hospital_A", "Govt_Hospital_B", "Private_Clinic_C"] if n != start_node]
        selected_stops = st.multiselect(
            "Healthcare destinations / service stops",
            vehicle_candidates,
            default=vehicle_candidates[:min(3, len(vehicle_candidates))],
            key="vehicle_stops",
        )
        vehicle_count = st.number_input(
            "Number of vehicles", min_value=1, max_value=max(1, len(selected_stops)), value=min(2, max(1, len(selected_stops))), step=1
        )
        if st.button("Optimize Vehicle Routes", type="secondary"):
            if len(selected_stops) < 2:
                st.warning("Select at least two destinations to demonstrate multi-vehicle optimization.")
            else:
                vrp = optimize_multi_vehicle(graph, selected_stops, int(vehicle_count))
                st.session_state["vrp_result"] = vrp

        vrp = st.session_state.get("vrp_result")
        if vrp:
            st.dataframe(vrp["routes"], use_container_width=True, hide_index=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total distance", f"{vrp['total_distance_km']} km")
            m2.metric("Max vehicle distance", f"{vrp['max_vehicle_distance_km']} km")
            m3.metric("Route balance gap", f"{vrp['balance_gap_km']} km")
            st.caption(
                f"Genetic Algorithm: {vrp['generations']} generations × {vrp['population_size']} candidates. "
                f"Fitness complexity: {vrp['complexity']}."
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
        st.dataframe(
            [
                {"Method": "Naive Backtracking", "Nodes Expanded": naive_result["nodes_expanded"], "Time (ms)": naive_result["time_taken_ms"]},
                {"Method": "Smart (MRV + Forward Checking)", "Nodes Expanded": smart_result["nodes_expanded"], "Time (ms)": smart_result["time_taken_ms"]},
            ],
            use_container_width=True, hide_index=True,
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
        st.dataframe(table_data, use_container_width=True, hide_index=True)

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
        for step in result["trace"]:
            indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * step["depth"]
            icon = "\u2713" if step["result"] else "\u2717"
            st.markdown(indent + icon + " " + str(step["goal"]) + " -- " + step["reason"], unsafe_allow_html=True)
