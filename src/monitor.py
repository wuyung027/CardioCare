import os
import json
import logging
from datetime import datetime, timedelta

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

from preprocessing import prepare_data


MODEL_PATH = "models/best_tuned_model.joblib"
REPORT_DIR = "reports"
LOG_DIR = "logs"

INFERENCE_LOG_PATH = os.path.join(LOG_DIR, "inference_monitor.log")
DRIFT_RESULT_PATH = os.path.join(REPORT_DIR, "drift_results.json")
PERFORMANCE_FIG_PATH = os.path.join(REPORT_DIR, "drift_performance_comparison.png")
TIME_SERIES_FIG_PATH = os.path.join(REPORT_DIR, "metric_time_series.png")

MODEL_VERSION = "best_tuned_model_v1"


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        filename=INFERENCE_LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        encoding="utf-8",
    )


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def calculate_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def create_drift_data(X_test):
    """
    실제 운영 중 데이터 분포 변화가 발생한 상황을 가정하기 위해
    일부 연속형 변수의 값을 인위적으로 이동시킨다.
    """
    X_drift = X_test.copy()

    X_drift["chol"] = X_drift["chol"] + 50
    X_drift["trestbps"] = X_drift["trestbps"] + 15
    X_drift["oldpeak"] = X_drift["oldpeak"] + 1.0

    return X_drift


def log_inference_records(model, X_data, y_true=None):
    """
    추론 시점의 핵심 정보를 logging으로 기록한다.
    """
    predictions = model.predict(X_data)

    for idx, pred in enumerate(predictions):
        actual_label = None
        if y_true is not None:
            actual_label = int(y_true.iloc[idx]) if hasattr(y_true, "iloc") else int(y_true[idx])

        logging.info(
            {
                "model_version": MODEL_VERSION,
                "input_shape": X_data.shape,
                "sample_index": int(idx),
                "prediction": int(pred),
                "actual_label": actual_label,
            }
        )

    return predictions


def run_ks_drift_test(X_train, X_drift, continuous_features):
    """
    학습 데이터 분포와 drift 데이터 분포를 비교하여 feature drift를 탐지한다.
    """
    drift_results = {}

    for feature in continuous_features:
        statistic, p_value = ks_2samp(X_train[feature], X_drift[feature])

        drift_results[feature] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05),
        }

    return drift_results


def save_performance_comparison(original_metrics, drift_metrics):
    metrics = ["balanced_accuracy", "precision", "recall", "f1"]

    original_values = [original_metrics[m] for m in metrics]
    drift_values = [drift_metrics[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], original_values, width, label="Original")
    plt.bar([i + width / 2 for i in x], drift_values, width, label="Drift")
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Original vs Drift Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PERFORMANCE_FIG_PATH)
    plt.close()


def save_metric_time_series(original_metrics, drift_metrics):
    """
    운영 시간이 지나며 성능이 낮아지는 상황을 보여주기 위한 synthetic time series graph.
    """
    start_date = datetime.now()

    dates = [start_date + timedelta(days=i) for i in range(6)]

    balanced_accuracy_values = [
        original_metrics["balanced_accuracy"],
        original_metrics["balanced_accuracy"] - 0.01,
        original_metrics["balanced_accuracy"] - 0.02,
        drift_metrics["balanced_accuracy"] + 0.03,
        drift_metrics["balanced_accuracy"] + 0.01,
        drift_metrics["balanced_accuracy"],
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(dates, balanced_accuracy_values, marker="o")
    plt.ylim(0, 1)
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Accuracy Over Time")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(TIME_SERIES_FIG_PATH)
    plt.close()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    setup_logging()

    model = load_trained_model()

    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Original test data evaluation
    original_predictions = log_inference_records(model, X_test, y_test)
    original_metrics = calculate_metrics(y_test, original_predictions)

    # Drift data evaluation
    X_drift = create_drift_data(X_test)
    drift_predictions = log_inference_records(model, X_drift, y_test)
    drift_metrics = calculate_metrics(y_test, drift_predictions)

    # Feature drift detection
    drift_results = run_ks_drift_test(
        X_train=X_train,
        X_drift=X_drift,
        continuous_features=continuous_features,
    )

    result = {
        "model_version": MODEL_VERSION,
        "original_metrics": original_metrics,
        "drift_metrics": drift_metrics,
        "ks_test_results": drift_results,
        "prediction_changes": int((original_predictions != drift_predictions).sum()),
        "total_samples": int(len(X_test)),
    }

    with open(DRIFT_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    save_performance_comparison(original_metrics, drift_metrics)
    save_metric_time_series(original_metrics, drift_metrics)

    print("Monitoring and drift detection completed.")
    print("=" * 60)
    print("Original metrics:")
    print(original_metrics)
    print("\nDrift metrics:")
    print(drift_metrics)
    print("\nKS-test results:")
    for feature, values in drift_results.items():
        print(
            f"{feature}: p-value={values['p_value']:.6f}, "
            f"drift_detected={values['drift_detected']}"
        )

    print(f"\nInference log saved to: {INFERENCE_LOG_PATH}")
    print(f"Drift results saved to: {DRIFT_RESULT_PATH}")
    print(f"Performance graph saved to: {PERFORMANCE_FIG_PATH}")
    print(f"Time series graph saved to: {TIME_SERIES_FIG_PATH}")


if __name__ == "__main__":
    main()