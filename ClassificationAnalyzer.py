#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    balanced_accuracy_score, recall_score, precision_score, f1_score)
from sklearn.preprocessing import label_binarize

class ClassificationAnalyzer:
    """
    Comprehensive classification model evaluator supporting:
    - Binary and multiclass metrics
    - ROC and Precision-Recall analysis
    - Threshold optimization (F1, Youden’s J)
    - Bootstrap confidence intervals
    - Interactive ROC curve
    """

    def __init__(self, model, X_test, y_test, labels=None, seed=0):
        self.model = model
        self.X_test = X_test
        self.y_test = np.array(y_test)
        self.labels = labels
        self.seed = seed

        self.y_proba = model.predict_proba(X_test)
        if self.y_proba.ndim == 1 or self.y_proba.shape[1] == 1:
            self.binary = True
            self.y_proba = self.y_proba.ravel()
        else:
            self.binary = False

    # ---------- Metric utilities ----------

    @staticmethod
    def specificity(y_true, y_pred):
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    @staticmethod
    def npv(y_true, y_pred):
        tn, _, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0

    @staticmethod
    def bootstrap_ci(y_true, y_input, metric_func, n_bootstrap=100, alpha=0.05, seed=0):
        np.random.seed(seed)
        stats = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            stats.append(metric_func(y_true[idx], y_input[idx]))
        lower = np.percentile(stats, 100 * (alpha / 2))
        upper = np.percentile(stats, 100 * (1 - alpha / 2))
        return (lower, upper)

    # ---------- Evaluation ----------

    def evaluate(self, threshold=0.5, bootstrap=False, n_bootstrap=500):
        """Evaluate metrics at a specific threshold."""
        if self.binary:
            y_pred = (self.y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            results = {
                "threshold": threshold,
                "accuracy": balanced_accuracy_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "specificity": self.specificity(self.y_test, y_pred),
                "npv": self.npv(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
                "auc": roc_auc_score(self.y_test, self.y_proba),
            }
        else:
            y_pred = np.argmax(self.y_proba, axis=1)
            results = {
                "accuracy": balanced_accuracy_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred, average="macro"),
                "precision": precision_score(self.y_test, y_pred, average="macro"),
                "f1": f1_score(self.y_test, y_pred, average="macro"),
                "auc": roc_auc_score(self.y_test, self.y_proba, multi_class="ovr"),
            }

        if bootstrap and self.binary:
            y_test_np = np.array(self.y_test)
            y_pred_np = np.array(y_pred)
            y_proba_np = np.array(self.y_proba)
            results["ci"] = {
                "accuracy": self.bootstrap_ci(y_test_np, y_pred_np, balanced_accuracy_score, n_bootstrap),
                "recall": self.bootstrap_ci(y_test_np, y_pred_np, recall_score, n_bootstrap),
                "precision": self.bootstrap_ci(y_test_np, y_pred_np, precision_score, n_bootstrap),
                "specificity": self.bootstrap_ci(y_test_np, y_pred_np, self.specificity, n_bootstrap),
                "npv": self.bootstrap_ci(y_test_np, y_pred_np, self.npv, n_bootstrap),
                "f1": self.bootstrap_ci(y_test_np, y_pred_np, f1_score, n_bootstrap),
                "auc": self.bootstrap_ci(y_test_np, y_proba_np, roc_auc_score, n_bootstrap),
            }

        return results

    # ---------- Threshold optimization ----------

    def optimize_threshold(self, metric="f1"):
        """Find optimal threshold based on F1 or Youden’s J."""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
        j_scores = tpr - fpr
        f1_scores = []
        for t in thresholds:
            y_pred = (self.y_proba >= t).astype(int)
            f1_scores.append(f1_score(self.y_test, y_pred))

        if metric.lower() == "youden":
            best_idx = np.argmax(j_scores)
        else:
            best_idx = np.argmax(f1_scores)

        return {
            "best_threshold": thresholds[best_idx],
            "best_score": f1_scores[best_idx] if metric == "f1" else j_scores[best_idx],
        }

    # ---------- Visualization ----------

    def plot_confusion_matrix(self, threshold=0.5, title="Confusion Matrix"):
        if self.binary:
            y_pred = (self.y_proba >= threshold).astype(int)
        else:
            y_pred = np.argmax(self.y_proba, axis=1)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"{title} (threshold={threshold:.2f})" if self.binary else title)
        plt.tight_layout()
        plt.show()

    def plot_precision_recall(self):
        if self.binary:
            precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall Curve")
            plt.show()
        else:
            y_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
            for i, label in enumerate(np.unique(self.y_test)):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], self.y_proba[:, i])
                plt.plot(recall, precision, label=f"Class {label}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Multiclass Precision–Recall Curve")
            plt.legend()
            plt.show()

    # ---------- Interactive ROC ----------

    def plot_interactive_roc(self):
        """Interactive ROC curve (binary & multiclass) with threshold hover info."""
        fig = go.Figure()

        if self.binary:
            fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
            auc = roc_auc_score(self.y_test, self.y_proba)

            accuracies, f1_scores = [], []
            for t in thresholds:
                y_pred = (self.y_proba >= t).astype(int)
                accuracies.append(balanced_accuracy_score(self.y_test, y_pred))
                f1_scores.append(f1_score(self.y_test, y_pred))

            hover_text = [
                f"Threshold: {th:.3f}<br>FPR: {fp:.3f}<br>TPR: {tp:.3f}<br>"
                f"F1: {f1:.3f}<br>Acc: {acc:.3f}"
                for th, fp, tp, f1, acc in zip(thresholds, fpr, tpr, f1_scores, accuracies)
            ]

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines+markers",
                text=hover_text, hoverinfo="text",
                name=f"ROC (AUC={auc:.3f})"
            ))

        else:
            y_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
            for i, label in enumerate(np.unique(self.y_test)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], self.y_proba[:, i])
                auc = roc_auc_score(y_bin[:, i], self.y_proba[:, i])
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"Class {label} (AUC={auc:.3f})"
                ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False
        ))

        fig.update_layout(
            title="Interactive ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            width=800, height=600,
            legend=dict(x=0.7, y=0.1)
        )
        fig.show()

    # ---------- Interactive Threshold Tuning ----------
    
    def plot_threshold_metrics(self):
        """Interactive plot showing F1, Precision, Recall, Accuracy vs threshold (binary only)."""
        if not self.binary:
            raise NotImplementedError("Interactive threshold plot is only implemented for binary classification.")

        thresholds = np.linspace(0, 1, 101)
        f1_scores, precisions, recalls, accuracies = [], [], [], []

        for t in thresholds:
            y_pred = (self.y_proba >= t).astype(int)
            f1_scores.append(f1_score(self.y_test, y_pred))
            precisions.append(precision_score(self.y_test, y_pred))
            recalls.append(recall_score(self.y_test, y_pred))
            accuracies.append(balanced_accuracy_score(self.y_test, y_pred))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode="lines+markers", name="F1 Score"))
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode="lines+markers", name="Precision"))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode="lines+markers", name="Recall"))
        fig.add_trace(go.Scatter(x=thresholds, y=accuracies, mode="lines+markers", name="Accuracy"))

        fig.update_layout(
            title=f"{self.model.__class__.__name__} Threshold Metrics",
            xaxis_title="Threshold",
            yaxis_title="Metric Value",
            template="plotly_white",
            width=800,
            height=500
        )
        fig.show()


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # ------------------ Generate synthetic binary data ------------------
    X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=0,
    n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ------------------ Train a classifier ------------------
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ------------------ Initialize ClassificationAnalyzer ------------------
    analyzer = ClassificationAnalyzer(clf, X_test, y_test, labels=[0, 1])

    # ------------------ Evaluate at default threshold ------------------
    results = analyzer.evaluate(threshold=0.5, bootstrap=True)
    print(results)

    # ------------------ Optimize threshold ------------------
    opt = analyzer.optimize_threshold(metric="f1")
    print(f"Optimal threshold (F1): {opt['best_threshold']:.2f}, score: {opt['best_score']:.3f}")

    # ------------------ Static plots ------------------
    analyzer.plot_confusion_matrix(threshold=0.5)
    analyzer.plot_precision_recall()

    # ------------------ Interactive plots ------------------
    analyzer.plot_interactive_roc()
    analyzer.plot_threshold_metrics()


