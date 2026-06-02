import os
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocessing import prepare_data


# ==============================
# 기본 설정
# ==============================

MODEL_DIR = "models"
REPORT_DIR = "reports"
RANDOM_STATE = 42

EXPERIMENT_NAME = "CardioCare_Heart_Disease"

# MLflow 기록 위치를 프로젝트 내부 mlruns 폴더로 고정
MLFLOW_TRACKING_URI = "file:./mlruns"


# ==============================
# 평가 함수
# ==============================

def evaluate_model(model, X_test, y_test):
    """
    학습된 모델을 테스트 데이터로 평가하는 함수
    """
    y_pred = model.predict(X_test)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm


def save_confusion_matrix(cm, model_name):
    """
    Confusion Matrix 이미지를 reports 폴더에 저장하는 함수
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"],
    )

    display.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    save_path = os.path.join(REPORT_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix 저장 완료: {save_path}")

    return save_path


def log_model_to_mlflow(model_name, pipeline, metrics, cm_path, cv_scores, extra_params=None):
    """
    모델 실험 결과를 MLflow에 기록하는 함수
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_family", model_name)

        if extra_params:
            mlflow.log_params(extra_params)

        # 테스트 평가 지표 기록
        mlflow.log_metric("balanced_accuracy", metrics["balanced_accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

        # 교차검증 결과 기록
        mlflow.log_metric("cv_balanced_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_balanced_accuracy_std", cv_scores.std())

        # Confusion Matrix 이미지 기록
        mlflow.log_artifact(cm_path)

        # 학습된 모델 기록
        mlflow.sklearn.log_model(pipeline, artifact_path="model")


# ==============================
# 기본 모델 3개 학습
# ==============================

def train_base_models():
    """
    Logistic Regression, SVC, Random Forest 3개 모델을 학습하고 비교하는 함수
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # MLflow 저장 위치와 실험명 설정
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 데이터 준비
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # 비교할 모델 3개
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "svc": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}

    best_model = None
    best_model_name = None
    best_score = -1

    for model_name, classifier in models.items():
        print("\n" + "=" * 60)
        print(f"모델 학습 시작: {model_name}")
        print("=" * 60)

        # 전처리 + 모델을 하나의 Pipeline으로 구성
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        # 5-fold 교차검증
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="balanced_accuracy",
        )

        print(f"5-fold CV balanced accuracy 평균: {cv_scores.mean():.4f}")
        print(f"5-fold CV balanced accuracy 표준편차: {cv_scores.std():.4f}")

        # 모델 학습
        pipeline.fit(X_train, y_train)

        # 테스트셋 평가
        metrics, cm = evaluate_model(pipeline, X_test, y_test)

        # Confusion Matrix 저장
        cm_path = save_confusion_matrix(cm, model_name)

        # 결과 저장
        results[model_name] = {
            "metrics": metrics,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"\n[{model_name} 평가 결과]")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        # MLflow 기록
        log_model_to_mlflow(
            model_name=model_name,
            pipeline=pipeline,
            metrics=metrics,
            cm_path=cm_path,
            cv_scores=cv_scores,
            extra_params=classifier.get_params(),
        )

        # 최종 모델은 balanced accuracy 기준으로 1차 선택
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            best_model = pipeline
            best_model_name = model_name

    # 기본 모델 중 최고 모델 저장
    best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_model, best_model_path)

    print("\n" + "=" * 60)
    print("기본 모델 비교 결과")
    print("=" * 60)
    print(f"기본 모델 중 최종 모델: {best_model_name}")
    print(f"Best Balanced Accuracy: {best_score:.4f}")
    print(f"모델 저장 위치: {best_model_path}")

    return X_train, X_test, y_train, y_test, preprocessor, results, best_model_name


# ==============================
# Logistic Regression 튜닝
# ==============================

def tune_logistic_regression(X_train, X_test, y_train, y_test, preprocessor):
    """
    Logistic Regression 하이퍼파라미터 C 값을 튜닝하는 함수
    """
    print("\n" + "=" * 60)
    print("Logistic Regression 하이퍼파라미터 튜닝 시작")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    # penalty 관련 경고를 줄이기 위해 C 값만 탐색
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_

    metrics, cm = evaluate_model(best_pipeline, X_test, y_test)
    cm_path = save_confusion_matrix(cm, "tuned_logistic_regression")

    print("\n[튜닝 결과]")
    print("Best Params:", grid_search.best_params_)
    print(f"Best CV Balanced Accuracy: {grid_search.best_score_:.4f}")

    print("\n[tuned_logistic_regression 테스트 평가 결과]")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # MLflow에 튜닝 모델 기록
    with mlflow.start_run(run_name="tuned_logistic_regression"):
        mlflow.set_tag("model_family", "logistic_regression_tuned")

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_balanced_accuracy", grid_search.best_score_)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

    # 튜닝 모델 저장
    tuned_model_path = os.path.join(MODEL_DIR, "best_tuned_model.joblib")
    joblib.dump(best_pipeline, tuned_model_path)

    print(f"\n튜닝 모델 저장 위치: {tuned_model_path}")

    return best_pipeline, metrics


# ==============================
# 결과 요약 저장
# ==============================

def save_results_summary(results):
    """
    모델 비교 결과를 JSON 파일로 저장하는 함수
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    serializable_results = {}

    for model_name, result in results.items():
        serializable_results[model_name] = {
            "metrics": {
                key: float(value)
                for key, value in result["metrics"].items()
            },
            "cv_mean": float(result["cv_mean"]),
            "cv_std": float(result["cv_std"]),
        }

    save_path = os.path.join(REPORT_DIR, "model_results_summary.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

    print(f"\n모델 비교 결과 저장 완료: {save_path}")


# ==============================
# 실행 시작점
# ==============================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, results, best_model_name = train_base_models()

    tuned_model, tuned_metrics = tune_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    save_results_summary(results)