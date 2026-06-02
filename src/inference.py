import os
import joblib
import pandas as pd


MODEL_PATH = "models/best_tuned_model.joblib"
SAMPLE_INPUT_PATH = "data/sample_input.csv"
PREDICTION_OUTPUT_PATH = "reports/predictions.csv"


def load_model(model_path=MODEL_PATH):
    """
    저장된 모델 파일을 불러오는 함수
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}\n"
            "먼저 python src/train.py 를 실행하여 모델을 학습하세요."
        )

    model = joblib.load(model_path)
    return model


def create_sample_input(save_path=SAMPLE_INPUT_PATH):
    """
    예측 테스트용 샘플 입력 파일을 생성하는 함수
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sample_data = pd.DataFrame(
        [
            {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 2,
                "thal": 3,
            },
            {
                "age": 41,
                "sex": 0,
                "cp": 1,
                "trestbps": 130,
                "chol": 204,
                "fbs": 0,
                "restecg": 0,
                "thalach": 172,
                "exang": 0,
                "oldpeak": 1.4,
                "slope": 2,
                "ca": 0,
                "thal": 2,
            },
        ]
    )

    sample_data.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"샘플 입력 파일 생성 완료: {save_path}")

    return sample_data


def load_input_data(input_path=SAMPLE_INPUT_PATH):
    """
    예측할 입력 데이터를 불러오는 함수
    """
    if not os.path.exists(input_path):
        print("샘플 입력 파일이 없어 새로 생성합니다.")
        return create_sample_input(input_path)

    input_data = pd.read_csv(input_path)
    return input_data


def validate_input_data(input_data):
    """
    입력 데이터의 기본 형식과 임상적 범위를 검증하는 함수

    테스트 코드에서 다음 항목을 확인하기 위해 사용한다.
    - 필수 컬럼 존재 여부
    - chol 값 범위
    - age 값 범위
    - trestbps 값 범위
    - thalach 값 범위
    """
    required_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")

    if not input_data["chol"].between(0, 600).all():
        raise ValueError("chol 값은 0 이상 600 이하이어야 합니다.")

    if not input_data["age"].between(0, 120).all():
        raise ValueError("age 값은 0 이상 120 이하이어야 합니다.")

    if not input_data["trestbps"].between(0, 250).all():
        raise ValueError("trestbps 값은 0 이상 250 이하이어야 합니다.")

    if not input_data["thalach"].between(0, 250).all():
        raise ValueError("thalach 값은 0 이상 250 이하이어야 합니다.")

    return True


def predict(model, input_data):
    """
    모델 예측을 수행하는 함수
    """
    validate_input_data(input_data)

    predictions = model.predict(input_data)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)
        disease_probabilities = probabilities[:, 1]
    else:
        disease_probabilities = [None] * len(predictions)

    result = input_data.copy()
    result["prediction"] = predictions
    result["disease_probability"] = disease_probabilities

    return result


def save_predictions(result, output_path=PREDICTION_OUTPUT_PATH):
    """
    예측 결과를 CSV 파일로 저장하는 함수
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"예측 결과 저장 완료: {output_path}")


def print_prediction_summary(result):
    """
    예측 결과를 터미널에 보기 좋게 출력하는 함수
    """
    print("\n예측 결과")
    print("=" * 60)

    for idx, row in result.iterrows():
        prediction = int(row["prediction"])
        probability = row["disease_probability"]

        label = "심장병 가능성 있음" if prediction == 1 else "정상 가능성 높음"

        print(f"\n샘플 {idx + 1}")
        print(f"예측값: {prediction} ({label})")

        if probability is not None:
            print(f"심장병 예측 확률: {probability:.4f}")


if __name__ == "__main__":
    model = load_model()
    input_data = load_input_data()
    result = predict(model, input_data)

    print_prediction_summary(result)
    save_predictions(result)