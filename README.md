# CardioCare

## 1. 프로젝트 개요

CardioCare는 심장병 관련 임상 데이터를 기반으로 환자의 심장병 가능성을 예측하는 End-to-End 머신러닝 프로젝트입니다.

본 프로젝트는 단순히 모델을 학습하는 것에 그치지 않고, 데이터 전처리, EDA, 모델 학습, MLflow 실험 관리, 하이퍼파라미터 튜닝, 추론, 단위 테스트, Docker 패키징, GitHub Actions CI, 데이터 드리프트 모니터링까지 포함하는 전체 머신러닝 파이프라인 구축을 목표로 합니다.

CardioCare는 의료진의 판단을 보조하기 위한 참고용 모델이며, 실제 진단이나 치료 결정을 대체하지 않습니다.

> Inform, not decide.
> CardioCare는 판단을 대신하는 것이 아니라, 판단에 참고할 수 있는 정보를 제공하는 것을 목표로 합니다.

---

## 2. 사용 데이터셋

본 프로젝트에서는 Heart Disease Dataset을 사용하였습니다.

* 데이터셋 이름: Heart Disease Dataset
* 문제 유형: 이진 분류(Binary Classification)
* 사용 목적: 환자 임상 정보를 기반으로 심장병 여부 예측
* 데이터 크기: 1025행, 14개 컬럼
* 중복 제거 후 데이터 크기: 302행

데이터는 다음 경로에 저장하여 사용합니다.

```text
data/heart.csv
```

주요 컬럼은 다음과 같습니다.

| 컬럼명        | 설명                 |
| ---------- | ------------------ |
| `age`      | 나이                 |
| `sex`      | 성별                 |
| `cp`       | 흉통 유형              |
| `trestbps` | 안정 혈압              |
| `chol`     | 콜레스테롤              |
| `fbs`      | 공복 혈당              |
| `restecg`  | 안정 심전도 결과          |
| `thalach`  | 최대 심박수             |
| `exang`    | 운동 유발 협심증 여부       |
| `oldpeak`  | 운동 후 ST depression |
| `slope`    | ST segment 기울기     |
| `ca`       | 주요 혈관 수            |
| `thal`     | thalassemia 관련 변수  |
| `target`   | 심장병 여부             |

target 값은 다음과 같이 해석합니다.

| target | 의미                    |
| ------ | --------------------- |
| 0      | No Heart Disease      |
| 1      | Heart Disease Present |

---

## 3. 프로젝트 구조

```text
CardioCare/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── heart.csv
│   └── sample_input.csv
├── models/
│   └── best_tuned_model.joblib
├── notebooks/
│   └── 01_eda_preprocessing.py
├── reports/
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_svc.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_tuned_logistic_regression.png
│   ├── continuous_features_boxplot.png
│   ├── target_distribution.png
│   ├── predictions.csv
│   ├── inference_log.csv
│   ├── drift_results.json
│   └── drift_performance_comparison.png
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 4. 개발 환경

본 프로젝트는 다음 환경을 기준으로 작성되었습니다.

* Python 3.11
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* scipy
* joblib
* mlflow
* Docker Desktop
* GitHub Actions

Docker 빌드 과정에서 Windows 전용 패키지로 인한 오류를 방지하기 위해 `requirements.txt`는 프로젝트 실행에 필요한 핵심 라이브러리 중심으로 정리하였습니다.

---

## 5. 설치 방법

프로젝트를 실행하기 위해 먼저 가상환경을 생성합니다.

```bash
python -m venv .venv
```

Windows 기준으로 가상환경을 실행합니다.

PowerShell:

```bash
.venv\Scripts\activate
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

---

## 6. EDA 및 전처리

EDA에서는 다음 항목을 확인하였습니다.

* 데이터 크기 확인
* `head()`, `info()`, `describe()` 확인
* target 분포 확인
* 결측값 확인
* 중복값 확인
* 연속형 변수 boxplot 확인

EDA 결과는 다음과 같습니다.

| 항목             | 결과          |
| -------------- | ----------- |
| 원본 데이터 크기      | 1025행, 14컬럼 |
| 중복 행 개수        | 723개        |
| 중복 제거 후 데이터 크기 | 302행        |
| 결측값            | 없음          |
| target 분포      | 비교적 균형적     |

전체 1025개 행 중 723개의 중복 행이 확인되었으므로, 학습 및 평가 과정에서 성능이 과대평가되는 것을 방지하기 위해 중복 행을 제거하였습니다.

전처리 단계에서는 입력 변수와 target을 분리한 뒤, 학습 데이터와 테스트 데이터를 8:2 비율로 나누었습니다. 또한 `stratify=y`를 사용하여 학습 데이터와 테스트 데이터의 target 비율이 유사하게 유지되도록 하였습니다.

중복 제거 후 데이터 분할 결과는 다음과 같습니다.

| 데이터     | 크기         |
| ------- | ---------- |
| X_train | 241행, 13컬럼 |
| X_test  | 61행, 13컬럼  |
| y_train | 241개       |
| y_test  | 61개        |

---

## 7. 모델 학습

본 프로젝트에서는 다음 세 가지 모델을 비교하였습니다.

* Logistic Regression
* Support Vector Classifier
* Random Forest Classifier

평가 지표는 다음을 사용하였습니다.

* Balanced Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

의료 예측 문제에서는 실제 심장병 환자를 정상으로 잘못 예측하는 False Negative가 중요하므로, 단순 정확도보다 recall과 balanced accuracy를 함께 고려하였습니다.

기본 모델 비교 결과는 다음과 같습니다.

| 모델                  | Balanced Accuracy | Precision | Recall |     F1 |
| ------------------- | ----------------: | --------: | -----: | -----: |
| Logistic Regression |            0.7992 |    0.8000 | 0.8485 | 0.8235 |
| SVC                 |            0.7841 |    0.7941 | 0.8182 | 0.8060 |
| Random Forest       |            0.7359 |    0.7576 | 0.7576 | 0.7576 |

---

## 8. 교차검증 및 하이퍼파라미터 튜닝

모델의 안정적인 성능 비교를 위해 5-fold cross validation을 수행하였습니다.

또한 Logistic Regression에 대해 `C` 값을 대상으로 GridSearchCV를 수행하였습니다.

튜닝 결과는 다음과 같습니다.

| 모델                        | Best Parameter | Balanced Accuracy | Precision | Recall |     F1 |
| ------------------------- | -------------- | ----------------: | --------: | -----: | -----: |
| Tuned Logistic Regression | `C=0.1`        |            0.8144 |    0.8056 | 0.8788 | 0.8406 |

최종 모델은 tuned Logistic Regression으로 선택하였습니다.

선택 이유는 다음과 같습니다.

* 테스트셋에서 가장 높은 balanced accuracy를 보임
* 심장병 예측 문제에서 중요한 recall이 가장 높음
* F1-score도 가장 안정적으로 나타남

최종 모델은 다음 경로에 저장하였습니다.

```text
models/best_tuned_model.joblib
```

---

## 9. MLflow 실험 관리

모델 학습 과정은 MLflow를 통해 기록하였습니다.

MLflow에는 다음 정보가 기록됩니다.

* 모델 이름
* 하이퍼파라미터
* Balanced Accuracy
* Precision
* Recall
* F1-score
* 5-fold CV 평균 및 표준편차
* Confusion Matrix 이미지
* 모델 artifact

MLflow UI 실행 방법은 다음과 같습니다.

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5001
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:5001
```

확인된 run은 다음과 같습니다.

* logistic_regression
* svc
* random_forest
* tuned_logistic_regression

---

## 10. 추론 실행

저장된 최종 모델을 불러와 샘플 입력 데이터에 대해 예측을 수행할 수 있습니다.

실행 명령어:

```bash
python src/inference.py
```

예상 출력 예시는 다음과 같습니다.

```text
예측 결과
============================================================

샘플 1
예측값: 0 (정상 가능성 높음)
심장병 예측 확률: 0.3030

샘플 2
예측값: 1 (심장병 가능성 있음)
심장병 예측 확률: 0.8549

예측 결과 저장 완료: reports/predictions.csv
```

예측 결과는 다음 파일에 저장됩니다.

```text
reports/predictions.csv
```

---

## 11. 단위 테스트

본 프로젝트는 `unittest`를 사용하여 머신러닝 파이프라인 테스트를 작성하였습니다.

테스트 항목은 다음과 같습니다.

1. 예측 결과의 행 개수가 입력 데이터의 행 개수와 일치하는지 확인
2. 예측 확률이 0 이상 1 이하인지 확인
3. 임상 입력값이 정상 범위일 때 검증을 통과하는지 확인
4. 비정상적인 `chol` 값 입력 시 오류가 발생하는지 확인
5. 동일한 입력 데이터에 대해 동일한 예측 결과가 나오는지 확인

테스트 실행 명령어:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

테스트 결과:

```text
Ran 5 tests

OK
```

---

## 12. Docker 실행

Dockerfile을 작성하여 프로젝트 실행 환경을 컨테이너로 패키징하였습니다.

Docker 이미지는 Python 3.11 slim 이미지를 기반으로 하며, `requirements.txt`를 통해 필요한 라이브러리를 설치한 뒤 `src/inference.py`를 실행하도록 구성하였습니다.

Docker 이미지 빌드:

```bash
docker build -t cardiocare:1.0 .
```

Docker 컨테이너 실행:

```bash
docker run cardiocare:1.0
```

실행 결과로 저장된 최종 모델이 샘플 입력 데이터에 대해 정상적으로 예측을 수행하는 것을 확인하였습니다.

---

## 13. GitHub Actions CI

GitHub Actions를 이용하여 CI 워크플로를 구성하였습니다.

`main` 브랜치에 push가 발생하면 GitHub Actions에서 다음 작업을 자동으로 수행합니다.

1. Ubuntu 환경에서 repository checkout
2. Python 3.11 설치
3. `requirements.txt` 기반 의존성 설치
4. unittest 자동 실행

CI 설정 파일은 다음 경로에 있습니다.

```text
.github/workflows/ci.yml
```

CI 실행 결과 최종적으로 테스트가 정상 통과하였습니다.

초기 실행에서는 모델 파일이 GitHub 저장소에 포함되지 않아 `FileNotFoundError`가 발생하였으나, 최종 모델 파일인 `models/best_tuned_model.joblib`을 저장소에 추가한 후 CI가 정상적으로 통과하였습니다.

---

## 14. 모니터링 및 데이터 드리프트 실험

모델 배포 이후 실제 서비스 환경에서는 입력 데이터의 분포가 학습 시점과 달라질 수 있습니다. 이를 데이터 드리프트라고 합니다.

본 프로젝트에서는 테스트 데이터의 `chol`과 `trestbps` 값을 인위적으로 증가시켜 환자군의 임상적 특성이 변화한 상황을 가정하였습니다.

실행 명령어:

```bash
python src/monitor.py
```

드리프트 실험 결과는 다음과 같습니다.

| 지표                | 원본 데이터 | Drift 데이터 |
| ----------------- | -----: | --------: |
| Balanced Accuracy | 0.8144 |    0.7413 |
| Precision         | 0.8056 |    0.7931 |
| Recall            | 0.8788 |    0.6970 |
| F1-score          | 0.8406 |    0.7419 |

KS 검정 결과는 다음과 같습니다.

| 변수       | KS Statistic |  p-value | Drift Detected |
| -------- | -----------: | -------: | -------------- |
| chol     |       0.4918 | 0.000000 | True           |
| trestbps |       0.4426 | 0.000009 | True           |

전체 61개 테스트 샘플 중 7개의 예측 결과가 drift 적용 후 변경되었습니다.

특히 심장병 예측 문제에서는 실제 심장병 환자를 정상으로 잘못 예측하는 False Negative를 줄이는 것이 중요하므로, drift 후 recall 감소는 운영 과정에서 주의 깊게 모니터링해야 할 지표입니다.

생성 파일은 다음과 같습니다.

```text
reports/inference_log.csv
reports/drift_results.json
reports/drift_performance_comparison.png
```

---

## 15. 실행 순서 요약

전체 프로젝트 실행 순서는 다음과 같습니다.

```bash
pip install -r requirements.txt
python src/train.py
python src/inference.py
python -m unittest discover -s tests -p "test_*.py"
python src/monitor.py
docker build -t cardiocare:1.0 .
docker run cardiocare:1.0
```

MLflow UI 확인:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5001
```

---

## 16. 윤리적 고려 사항

CardioCare는 심장병 예측을 위한 보조 도구입니다.

이 모델의 예측 결과는 의료진의 판단을 보조하기 위한 참고 자료로만 사용되어야 하며, 환자에 대한 최종 진단이나 치료 결정은 반드시 전문 의료진이 수행해야 합니다.

특히 심장병 예측 문제에서는 실제 심장병 환자를 정상으로 잘못 예측하는 False Negative가 치명적일 수 있으므로, 모델 평가 시 단순 정확도뿐 아니라 recall, balanced accuracy, confusion matrix를 함께 고려하였습니다.

---

## 17. 한계점 및 개선 방향

본 프로젝트는 공개 데이터셋을 기반으로 한 학습용 프로젝트이므로 실제 의료 환경에 바로 적용하기에는 한계가 있습니다.

주요 한계는 다음과 같습니다.

* 데이터셋의 크기가 제한적임
* 중복 제거 후 학습 데이터가 302개로 줄어듦
* 실제 병원 환경의 다양한 환자군을 모두 반영하지 못할 수 있음
* 공개 데이터 기반 모델이므로 의학적 검증이 별도로 필요함
* 데이터 드리프트 발생 시 recall이 크게 감소할 수 있음

추후 개선 방향은 다음과 같습니다.

* 더 큰 규모의 실제 임상 데이터 적용
* SHAP 등을 활용한 모델 설명 가능성 강화
* API 기반 모델 서빙 구조 추가
* 주기적인 재학습 자동화
* drift 감지 후 재학습 트리거 구조 설계

---

## 18. AI 도구 사용 공개

본 프로젝트를 진행하는 과정에서 AI 도구를 일부 활용하였습니다.

AI 도구는 다음 목적으로 활용하였습니다.

* 프로젝트 구조 설계 보조
* 코드 작성 방향 검토
* 오류 메시지 분석
* MLflow tracking URI 문제 해결
* Docker 빌드 오류 원인 분석
* README 및 보고서 초안 작성 보조
* 머신러닝 개념 이해 보조

특히 MLflow UI에서 실험 기록이 `Default`만 표시되는 문제가 발생했을 때, AI 도구를 활용하여 tracking URI 설정 문제와 터미널 환경 문제를 점검하였습니다. 이후 `mlflow.set_tracking_uri("file:./mlruns")`를 명시하고 Command Prompt에서 재실행하여 `CardioCare_Heart_Disease` 실험 기록을 정상적으로 확인하였습니다.

또한 Docker 빌드 과정에서 Windows 전용 패키지인 `pywin32`가 Linux 기반 Docker 컨테이너에서 설치되지 않는 문제가 발생하였습니다. AI 도구를 활용해 오류 원인을 확인한 뒤, 프로젝트 실행에 필요한 핵심 라이브러리만 남기도록 `requirements.txt`를 정리하여 Docker 빌드 문제를 해결하였습니다.

단, 최종 제출 코드와 결과는 작성자가 직접 실행하고 검토하였습니다.
