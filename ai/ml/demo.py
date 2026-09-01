"""
ML Module Demo -- Decision Tree + K-Means
===============================================

Runs both ML components back to back and prints a combined summary,
matching what you'd present in your report/viva for Module 7 and 8.

Usage:
    cd ai/ml
    python demo.py
"""

from decision_tree import train_and_evaluate, predict_service
from kmeans import run_kmeans, profile_clusters, label_cluster, FEATURE_NAMES as EXP_FEATURES


def demo_decision_tree():
    print("=" * 78)
    print("MODULE 7 -- Decision Tree: Healthcare Service Category Classification")
    print("=" * 78)
    results = train_and_evaluate()

    print(f"Train / Test split: {results['n_train']} / {results['n_test']} samples")
    print(f"Accuracy: {results['accuracy']:.3f}\n")
    print(results["classification_report"])

    print("Top features driving the decision:")
    top_features = sorted(results["feature_importances"].items(), key=lambda x: -x[1])[:3]
    for feat, importance in top_features:
        print(f"  {feat}: {importance:.3f}")

    example_patient = {
        "age": 25, "fever_severity_score": 0, "fever_duration_days": 0,
        "chronic_condition": 0, "needs_vaccination": 1, "mobility_impaired": 0,
        "prior_visits_count": 5,
    }
    predicted = predict_service(results["model"], example_patient)
    print(f"\nExample: 25-year-old requesting vaccination, no symptoms -> predicted: {predicted}")

    print(
        "\nNote: at only ~92% accuracy, some edge-case combinations near a "
        "decision boundary WILL get misclassified -- e.g. a first-time "
        "vaccination request with very few prior visits can occasionally "
        "be predicted as Urgent_OPD instead, because that specific leaf "
        "of the tree has very few training examples and is sensitive to "
        "the 5% label noise injected in the synthetic dataset. This is a "
        "real and worth-mentioning limitation in your report: small "
        "leaves in a Decision Tree are more vulnerable to noisy labels "
        "than leaves with lots of supporting examples."
    )


def demo_kmeans():
    print("\n" + "=" * 78)
    print("MODULE 8 -- K-Means: Patient Experience Segmentation")
    print("=" * 78)
    result = run_kmeans()
    print(f"Silhouette score: {result['silhouette_score']:.3f}\n")

    profiles = profile_clusters(result["rows"], result["k"])
    for cluster_id, profile in profiles.items():
        label = label_cluster(profile)
        print(f"Cluster {cluster_id} -- {label} (n={profile['count']})")

    # ---- The "Experience Optimization" insight the roadmap describes ----
    worst_cluster_id = max(
        profiles, key=lambda cid: profiles[cid]["waiting_time_minutes"]
    )
    worst = profiles[worst_cluster_id]
    print(
        f"\nOptimization insight: Cluster {worst_cluster_id} "
        f"('{label_cluster(worst)}') has the highest average wait "
        f"({worst['waiting_time_minutes']} min) and average satisfaction of only "
        f"{worst['satisfaction_score']}/10 -- these users are the best target "
        f"for process improvements (e.g. pre-booking, SMS queue updates, "
        f"routing to a less congested facility when capacity allows)."
    )


if __name__ == "__main__":
    demo_decision_tree()
    demo_kmeans()
