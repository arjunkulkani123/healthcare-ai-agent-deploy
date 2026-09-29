"""
K-Means -- Patient Experience Segmentation
================================================

Clusters patients by their post-visit EXPERIENCE metrics (wait time,
navigation friction, travel distance, digital usage, satisfaction,
visit history) to find natural groupings -- this is the roadmap's
"Experience Optimization" analysis (Section 13), feeding into the
Optimization Agent's "users like X experience worse service in area Y"
kind of insight.

Evaluation follows Section 21: not just "we ran K-Means" -- includes
silhouette score (cluster separation quality) and a human-readable
profile of what each cluster represents.

Requires scikit-learn (see requirements.txt).
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from dataset_experience import generate_dataset, FEATURE_NAMES


def load_X(n_samples: int = 400, seed: int = 7):
    rows = generate_dataset(n_samples=n_samples, seed=seed)
    X = [[row[f] for f in FEATURE_NAMES] for row in rows]
    return X, rows


def run_kmeans(n_samples: int = 400, k: int = 4, seed: int = 7):
    X, rows = load_X(n_samples=n_samples, seed=seed)

    # Standardize features first -- K-Means uses Euclidean distance, and
    # our features have very different scales (minutes vs a 0-10 score),
    # so without scaling, waiting_time_minutes would dominate the
    # distance calculation and drown out the other features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, cluster_labels)

    # Attach cluster assignment back to the original (unscaled) rows so
    # the profile summary below is in human-readable units.
    for row, label in zip(rows, cluster_labels):
        row["cluster"] = int(label)

    return {
        "rows": rows,
        "cluster_labels": cluster_labels,
        "silhouette_score": sil_score,
        "model": model,
        "scaler": scaler,
        "k": k,
    }


def profile_clusters(rows: list, k: int) -> dict:
    """Average feature values per cluster, to interpret what each one represents."""
    profiles = {}
    for cluster_id in range(k):
        members = [r for r in rows if r["cluster"] == cluster_id]
        if not members:
            continue
        profile = {"count": len(members)}
        for feat in FEATURE_NAMES:
            profile[feat] = round(sum(m[feat] for m in members) / len(members), 2)
        profiles[cluster_id] = profile
    return profiles


def label_cluster(profile: dict) -> str:
    """Heuristic human-readable label based on the cluster's average metrics."""
    if profile["satisfaction_score"] >= 7 and profile["waiting_time_minutes"] < 25:
        return "Highly satisfied users"
    if profile["waiting_time_minutes"] >= 45:
        return "Long-wait users"
    if profile["navigation_steps"] >= 7 or profile["travel_distance_km"] >= 8:
        return "High-friction users"
    if profile["digital_usage_score"] <= 2:
        return "Digitally underserved users"
    return "Mixed/moderate-experience users"


if __name__ == "__main__":
    result = run_kmeans()
    print(f"Silhouette score: {result['silhouette_score']:.3f}  "
          f"(closer to 1 = well-separated clusters, near 0 = overlapping)\n")

    profiles = profile_clusters(result["rows"], result["k"])
    for cluster_id, profile in profiles.items():
        label = label_cluster(profile)
        print(f"Cluster {cluster_id} -- {label}  (n={profile['count']})")
        for feat in FEATURE_NAMES:
            print(f"    {feat:22s}: {profile[feat]}")
        print()
