"""
Decision Tree -- Healthcare Service Category Classification
================================================================

Trains a Decision Tree to predict which SERVICE CATEGORY (Emergency,
Urgent_OPD, Vaccination, Specialist_Consultation, General_OPD) a patient
should be routed to, based on structured intake features.

Evaluation follows the roadmap's requirement (Section 21): not just a
bare accuracy number, but train/test split, precision/recall/F1 per
class, and a confusion matrix.

Requires scikit-learn (see requirements.txt):
    pip install -r requirements.txt
"""

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset_classification import generate_dataset, FEATURE_NAMES, LABEL_NAME


def load_xy(n_samples: int = 600, seed: int = 42):
    rows = generate_dataset(n_samples=n_samples, seed=seed)
    X = [[row[f] for f in FEATURE_NAMES] for row in rows]
    y = [row[LABEL_NAME] for row in rows]
    return X, y


def train_and_evaluate(n_samples: int = 600, test_size: float = 0.25, max_depth: int = 5, seed: int = 42):
    X, y = load_xy(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {
        "model": clf,
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=sorted(set(y))),
        "labels": sorted(set(y)),
        "feature_importances": dict(zip(FEATURE_NAMES, clf.feature_importances_)),
        "tree_text": export_text(clf, feature_names=FEATURE_NAMES),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return results


def predict_service(clf, patient: dict) -> str:
    """Predict a single patient's service category from a feature dict."""
    x = [[patient[f] for f in FEATURE_NAMES]]
    return clf.predict(x)[0]


if __name__ == "__main__":
    results = train_and_evaluate()

    print(f"Train size: {results['n_train']}, Test size: {results['n_test']}\n")
    print(f"Accuracy: {results['accuracy']:.3f}\n")

    print("Classification report (precision / recall / F1 per class):")
    print(results["classification_report"])

    print("Confusion matrix (rows = true label, cols = predicted):")
    print("Labels order:", results["labels"])
    for row in results["confusion_matrix"]:
        print(row)

    print("\nFeature importances:")
    for feat, importance in sorted(results["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"  {feat:25s}: {importance:.3f}")

    print("\nDecision tree structure (text form):")
    print(results["tree_text"])

    # Example single prediction
    example_patient = {
        "age": 65, "fever_severity_score": 3, "fever_duration_days": 1,
        "chronic_condition": 0, "needs_vaccination": 0, "mobility_impaired": 0,
        "prior_visits_count": 2,
    }
    predicted = predict_service(results["model"], example_patient)
    print(f"\nExample prediction for a 65-year-old with a high 1-day fever: {predicted}")
