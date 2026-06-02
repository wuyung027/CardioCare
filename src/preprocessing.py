import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


DATA_PATH = "data/heart.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def load_data(path=DATA_PATH):
    """
    heart.csv 파일을 불러오는 함수
    """
    df = pd.read_csv(path)
    return df


def binarize_target(df, target_col="target"):
    """
    target이 다중 클래스인 경우 이진 분류 형태로 변환하는 함수

    target = 0  -> 정상
    target > 0  -> 심장병 가능성 있음
    """
    df = df.copy()
    df[target_col] = (df[target_col] > 0).astype(int)
    return df


def remove_duplicates(df):
    """
    중복 행을 제거하는 함수
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"중복 제거 전 데이터 수: {before}")
    print(f"중복 제거 후 데이터 수: {after}")
    print(f"제거된 중복 행 수: {before - after}")

    return df


def split_features_target(df, target_col="target"):
    """
    입력 변수 X와 정답 y를 분리하는 함수
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor():
    """
    전처리 파이프라인을 생성하는 함수

    - 연속형 변수: 결측값은 중앙값으로 대체, StandardScaler 적용
    - 범주형 변수: 결측값은 최빈값으로 대체
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def prepare_data(path=DATA_PATH):
    """
    전체 데이터 준비 함수

    1. 데이터 불러오기
    2. target 이진화
    3. 중복 제거
    4. X, y 분리
    5. train/test split
    6. 전처리 파이프라인 생성
    """
    df = load_data(path)
    df = binarize_target(df)
    df = remove_duplicates(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    print("\n데이터 준비 완료")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\n학습 데이터 target 분포:")
    print(y_train.value_counts(normalize=True))

    print("\n테스트 데이터 target 분포:")
    print(y_test.value_counts(normalize=True))