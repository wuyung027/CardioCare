import os
import json
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from preprocessing import prepare_data


REPORT_DIR = "reports"
FEATURE_SELECTION_RESULT_PATH = "reports/selected_features.json"


def clean_feature_name(feature_name):
    """
    ColumnTransformer에서 생성된 feature 이름의 prefix를 제거하는 함수
    예: numeric__age -> age
    """
    if "__" in feature_name:
        return feature_name.split("__")[-1]
    return feature_name


def run_feature_selection():
    """
    Random Forest 기반 SelectFromModel을 사용하여 특성 선택을 수행한다.
    특성 선택은 train/test split 이후 학습 데이터에 대해서만 수행하여 데이터 누수를 방지한다.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # 전처리 기준은 학습 데이터에만 fit
    X_train_processed = preprocessor.fit_transform(X_train)

    # 전처리 후 feature 이름 추출
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in feature_names]

    # Random Forest 기반 특성 중요도 계산
    rf_selector_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    selector = SelectFromModel(
        estimator=rf_selector_model,
        threshold="median",
    )

    selector.fit(X_train_processed, y_train)

    selected_mask = selector.get_support()
    selected_features = [
        feature
        for feature, selected in zip(feature_names, selected_mask)
        if selected
    ]

    importances = selector.estimator_.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "selected": selected_mask,
        }
    ).sort_values(by="importance", ascending=False)

    result = {
        "method": "RandomForestClassifier + SelectFromModel",
        "threshold": "median",
        "selected_features": selected_features,
        "all_feature_importances": feature_importance_df.to_dict(orient="records"),
    }

    with open(FEATURE_SELECTION_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("\n특성 선택 완료")
    print("=" * 60)
    print(f"선택된 특성 개수: {len(selected_features)} / {len(feature_names)}")
    print("\n선택된 특성:")
    for feature in selected_features:
        print(f"- {feature}")

    print(f"\n특성 선택 결과 저장 완료: {FEATURE_SELECTION_RESULT_PATH}")
    print("\n전체 특성 중요도:")
    print(feature_importance_df)


if __name__ == "__main__":
    run_feature_selection()