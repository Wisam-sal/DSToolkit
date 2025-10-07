
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

class RegressionAnalyzer:
    """Regression model performance, diagnostics, and error analysis."""

    def __init__(self, y_test, y_pred, model_name="Model"):
        self.y_test = np.asarray(y_test)
        self.y_pred = np.asarray(y_pred)
        self.model_name = model_name
        self.residuals = self.y_test - self.y_pred

        self.metrics = self._compute_metrics()
        self.normality_p = self._shapiro_test()

    # ---------------- Metrics ---------------- #
    def _compute_metrics(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
            "Residual Mean": np.mean(self.residuals),
            "Residual Std": np.std(self.residuals),
        }

    def _shapiro_test(self):
        _, p_value = shapiro(self.residuals)
        return p_value

    # ---------------- Plotting ---------------- #
    def plot_residual_histogram(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(self.residuals, bins=30, kde=True, ax=ax, color="#1f77b4")
        ax.axvline(0, color="k", linestyle="--", lw=1)
        ax.set_title(f"{self.model_name} | Residuals Distribution", fontsize=10, pad=8)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        sns.despine()
        return ax

    def plot_qq(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        probplot(self.residuals, dist="norm", plot=ax)
        ax.set_title(f"{self.model_name} | Residuals Q-Q Plot", fontsize=10, pad=8)
        sns.despine()
        return ax

    def plot_predicted_vs_actual(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(x=self.y_test, y=self.y_pred, ax=ax, color="#9467bd", alpha=0.7, edgecolor=None)
        ax.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--", lw=1
        )
        ax.set_title(f"{self.model_name} | Predicted vs Actual", fontsize=10, pad=8)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        sns.despine()
        return ax

    def plot_all(self):
        fig, axs = plt.subplots(3, 1, figsize=(6, 10))
        self.plot_residual_histogram(axs[0])
        self.plot_qq(axs[1])
        self.plot_predicted_vs_actual(axs[2])
        fig.suptitle(f"{self.model_name} | Regression Diagnostics", fontsize=12, weight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    # ---------------- Summary ---------------- #
    def summary(self):
        border = "=" * 65
        print(border)
        print(f"{'Model Summary':^65}")
        print(border)
        print(f"Model Name: {self.model_name}")
        for k, v in self.metrics.items():
            print(f"{k:<20}: {v:>10.4f}")
        print(f"Shapiro-Wilk p-value : {self.normality_p:.6f} "
              f"({'Normal' if self.normality_p > 0.05 else 'Non-normal'})")
        print(border)