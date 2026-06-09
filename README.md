# CardioCare

## 1. 프로젝트 개요

CardioCare는 심장병 관련 임상 데이터를 기반으로 환자의 심장병 가능성을 예측하는 End-to-End 머신러닝 프로젝트입니다.

본 프로젝트는 단순히 모델을 학습하는 것에 그치지 않고, 데이터 전처리, EDA, 특성 선택, 모델 학습, MLflow 실험 관리, 하이퍼파라미터 튜닝, 추론, 단위 테스트, Docker 패키징, GitHub Actions CI, 데이터 드리프트 모니터링까지 포함하는 전체 머신러닝 파이프라인 구축을 목표로 합니다.

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

`target` 값은 다음과 같이 해석합니다.

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
├── logs/
│   └── inference_monitor.log
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
│   ├── selected_features.json
│   ├── drift_results.json
│   ├── drift_performance_comparison.png
│   └── metric_time_series.png
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
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

전처리 파이프라인에서는 연속형 변수와 범주형 변수를 구분하였습니다. 연속형 변수에는 `StandardScaler`를 적용하였고, 결측값 발생 가능성을 고려하여 연속형 변수는 중앙값, 범주형 변수는 최빈값으로 대체하도록 구성하였습니다. 전처리 과정은 `ColumnTransformer`와 `Pipeline`을 사용하여 학습, 평가, 추론 단계에서 동일하게 재사용할 수 있도록 하였습니다.

---

## 7. 특성 선택

모델 학습에 앞서 Random Forest 기반 `SelectFromModel`을 사용하여 특성 선택을 수행하였습니다. 특성 선택은 데이터 누수를 방지하기 위해 train/test split 이후 학습 데이터에 대해서만 수행하였습니다.

선택된 특성은 다음과 같습니다.

| 선택된 특성    | 설명                 |
| --------- | ------------------ |
| `age`     | 나이                 |
| `chol`    | 콜레스테롤              |
| `thalach` | 최대 심박수             |
| `oldpeak` | 운동 후 ST depression |
| `cp`      | 흉통 유형              |
| `ca`      | 주요 혈관 수            |
| `thal`    | thalassemia 관련 변수  |

특성 선택 결과는 다음 파일에 저장됩니다.

```text
reports/selected_features.json
```

다만 최종 모델 학습에서는 데이터 크기가 제한적인 점을 고려하여 전체 13개 입력 변수를 사용하였습니다. 특성 선택 결과는 모델 해석과 변수 중요도 확인을 위한 보조 자료로 활용하였습니다.

특성 선택 실행 명령어는 다음과 같습니다.

```bash
python src/feature_selection.py
```

---

## 8. 모델 학습

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

| 모델                  | Balanced Accuracy | Precision | Recall | F1-score |
| ------------------- | ----------------: | --------: | -----: | -------: |
| Logistic Regression |            0.7992 |    0.8000 | 0.8485 |   0.8235 |
| SVC                 |            0.7841 |    0.7941 | 0.8182 |   0.8060 |
| Random Forest       |            0.7359 |    0.7576 | 0.7576 |   0.7576 |

---

## 9. 교차검증 및 하이퍼파라미터 튜닝

모델의 안정적인 성능 비교를 위해 5-fold cross validation을 수행하였습니다.

또한 Logistic Regression에 대해 `C` 값을 대상으로 `GridSearchCV`를 수행하였습니다.

튜닝 결과는 다음과 같습니다.

| 모델                        | Best Parameter | Balanced Accuracy | Precision | Recall | F1-score |
| ------------------------- | -------------- | ----------------: | --------: | -----: | -------: |
| Tuned Logistic Regression | `C=0.1`        |            0.8144 |    0.8056 | 0.8788 |   0.8406 |

최종 모델은 tuned Logistic Regression으로 선택하였습니다.

선택 이유는 다음과 같습니다.

* 테스트셋에서 가장 높은 balanced accuracy를 보임
* 심장병 예측 문제에서 중요한 recall이 가장 높음
* F1-score도 가장 안정적으로 나타남
* Logistic Regression은 비교적 해석이 쉬워 의료 보조 모델로 설명하기 적합함

최종 모델은 다음 경로에 저장하였습니다.

```text
models/best_tuned_model.joblib
```

---

## 10. MLflow 실험 관리

모델 학습 과정은 MLflow를 통해 기록하였습니다.

MLflow에는 다음 정보가 기록됩니다.

* 모델 이름
* 모델 계열 정보
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

## 11. 추론 실행

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

## 12. 단위 테스트

본 프로젝트는 `unittest`를 사용하여 머신러닝 파이프라인 테스트를 작성하였습니다.

테스트 항목은 다음과 같습니다.

1. 예측 결과의 행 개수가 입력 데이터의 행 개수와 일치하는지 확인
2. 예측 확률이 0 이상 1 이하인지 확인
3. 각 샘플에 대한 클래스별 예측 확률의 합이 1에 가까운지 확인
4. 임상 입력값이 정상 범위일 때 검증을 통과하는지 확인
5. 비정상적인 `chol` 값 입력 시 `ValueError`가 발생하는지 확인
6. 동일한 입력 데이터에 대해 동일한 예측 결과가 나오는지 확인

테스트 실행 명령어:

```bash
python -m unittest tests/test_pipeline.py
```

테스트 결과:

```text
Ran 6 tests

OK
```

---

## 13. Docker 실행

`Dockerfile`을 작성하여 프로젝트 실행 환경을 컨테이너로 패키징하였습니다.

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

## 14. GitHub Actions CI

GitHub Actions를 이용하여 CI 워크플로를 구성하였습니다.

`main` 브랜치에 push가 발생하면 GitHub Actions에서 다음 작업을 자동으로 수행합니다.

1. Ubuntu 환경에서 repository checkout
2. Python 3.11 설치
3. `requirements.txt` 기반 의존성 설치
4. `unittest` 자동 실행

CI 설정 파일은 다음 경로에 있습니다.

```text
.github/workflows/ci.yml
```

CI 실행 결과 최종적으로 테스트가 정상 통과하였습니다.

초기 실행에서는 모델 파일이 GitHub 저장소에 포함되지 않아 `FileNotFoundError`가 발생하였으나, 최종 모델 파일인 `models/best_tuned_model.joblib`을 저장소에 추가한 후 CI가 정상적으로 통과하였습니다.

---

## 15. 모니터링 및 데이터 드리프트 실험

모델 배포 이후 실제 서비스 환경에서는 입력 데이터의 분포가 학습 시점과 달라질 수 있습니다. 이를 데이터 드리프트라고 합니다.

본 프로젝트에서는 `src/monitor.py`를 작성하여 추론 로그 기록과 drift detection을 수행하였습니다. 추론 과정에서는 `logging`을 사용하여 timestamp, model version, input shape, prediction, actual label을 `logs/inference_monitor.log` 파일에 기록합니다.

드리프트 상황을 실험하기 위해 테스트 데이터의 `chol`, `trestbps`, `oldpeak` 값을 인위적으로 증가시켜 환자군의 임상적 특성이 변화한 상황을 가정하였습니다.

실행 명령어:

```bash
python src/monitor.py
```

드리프트 실험 결과는 다음과 같습니다.

| 지표                | 원본 데이터 | Drift 데이터 |
| ----------------- | -----: | --------: |
| Balanced Accuracy | 0.8144 |    0.7646 |
| Precision         | 0.8056 |    0.8750 |
| Recall            | 0.8788 |    0.6364 |
| F1-score          | 0.8406 |    0.7368 |

KS 검정 결과는 다음과 같습니다.

| 변수         |  p-value | Drift Detected |
| ---------- | -------: | -------------- |
| `age`      | 0.239621 | False          |
| `trestbps` | 0.000000 | True           |
| `chol`     | 0.000000 | True           |
| `thalach`  | 0.408505 | False          |
| `oldpeak`  | 0.000000 | True           |

`trestbps`, `chol`, `oldpeak`에서 p-value가 0.05보다 작게 나타나 drift가 탐지되었습니다. 또한 drift 데이터에서는 recall이 0.8788에서 0.6364로 감소하였습니다. 심장병 예측 문제에서는 실제 심장병 환자를 정상으로 잘못 예측하는 False Negative를 줄이는 것이 중요하므로, drift 이후 recall 감소는 운영 과정에서 주의 깊게 모니터링해야 할 지표입니다.

생성 파일은 다음과 같습니다.

```text
logs/inference_monitor.log
reports/drift_results.json
reports/drift_performance_comparison.png
reports/metric_time_series.png
```

---

## 16. 실행 순서 요약

전체 프로젝트 실행 순서는 다음과 같습니다.

```bash
pip install -r requirements.txt
python src/train.py
python src/feature_selection.py
python src/inference.py
python -m unittest tests/test_pipeline.py
python src/monitor.py
docker build -t cardiocare:1.0 .
docker run cardiocare:1.0
```

MLflow UI 확인:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5001
```

---

## 17. 재학습 및 피드백 루프 전략

실제 서비스 환경에서는 새로운 환자 데이터가 계속 들어올 수 있으며, 시간이 지나면서 환자군의 특성이나 측정 환경이 달라질 수 있습니다. 따라서 모델의 성능을 안정적으로 유지하기 위해서는 주기적인 모니터링과 재학습 구조가 필요합니다.

재학습 전략은 다음과 같이 설계할 수 있습니다.

1. 운영 중 들어오는 입력 데이터의 주요 변수 분포를 기록
2. 학습 시점의 기준 데이터와 최근 입력 데이터의 분포 비교
3. KS 검정이나 PSI와 같은 통계적 방법으로 drift 확인
4. recall, balanced accuracy, confusion matrix 등 성능 지표 확인
5. 성능 저하 또는 drift 반복 감지 시 재학습 후보로 판단
6. 새 데이터와 기존 데이터를 함께 사용하여 모델 재학습
7. 기존 모델과 새 모델의 성능 비교
8. 새 모델이 더 안정적인 성능을 보일 경우에만 교체

다만 의료 예측 모델에서는 drift가 탐지되었다고 해서 자동으로 재학습 후 즉시 배포하는 방식은 적절하지 않습니다. 새로 수집된 데이터에 오류가 있거나 특정 환자군의 데이터가 과도하게 반영될 경우, 모델이 잘못된 방향으로 반복 학습되는 runaway feedback 문제가 발생할 수 있기 때문입니다. 따라서 Human-in-the-loop 방식을 적용하여 개발자 또는 의료 전문가가 데이터 품질, 라벨 정확성, 성능 변화 원인을 검토한 뒤 재학습 및 모델 교체 여부를 결정해야 합니다.

---

## 18. 서빙 방식 선택

본 프로젝트에서는 최종 모델을 `models/best_tuned_model.joblib` 파일로 저장하고, `src/inference.py`를 통해 저장된 모델을 불러와 예측을 수행하는 Docker 기반 배치 추론 방식을 구현하였습니다.

현재 구현 단계에서는 `docker run cardiocare:1.0` 명령어를 통해 저장된 모델을 불러오고, 샘플 입력 데이터에 대한 예측 결과와 심장병 예측 확률을 출력하도록 구성하였습니다. 이 방식은 과제 수준에서 모델이 독립된 환경에서도 재현 가능하게 실행되는지를 확인하는 데 적합합니다.

향후 실제 서비스로 확장한다면 MaaS(Model as a Service) 방식이 적합하다고 판단하였습니다. MaaS 방식은 모델을 서버 또는 API 형태로 배포하고, 사용자가 입력한 환자 정보를 서버에 전달하면 서버에서 예측 결과를 반환하는 구조입니다.

본 프로젝트의 심장병 예측 모델은 실시간 응급 처치 판단보다는 진료 보조를 위한 위험도 예측에 가깝기 때문에, 매우 낮은 지연시간이 필요한 On-Device 방식보다는 서버 기반 MaaS 방식으로도 충분히 활용 가능하다고 판단하였습니다. 또한 MaaS 방식은 서버에 배포된 모델만 교체하면 되므로, 여러 사용자 기기에 각각 모델을 다시 배포해야 하는 On-Device 방식보다 모델 업데이트와 유지보수가 쉽습니다.

다만 의료 데이터는 PHI와 같은 민감한 개인정보를 포함할 수 있으므로, MaaS 방식에서는 데이터 전송과 저장 과정의 보안 관리가 중요합니다. 실제 운영 환경에서는 개인정보 비식별화, 암호화, 접근 권한 관리, 로그 보관 정책이 함께 적용되어야 합니다.

---

## 19. 윤리적 고려 사항

CardioCare는 심장병 예측을 위한 보조 도구입니다.

이 모델의 예측 결과는 의료진의 판단을 보조하기 위한 참고 자료로만 사용되어야 하며, 환자에 대한 최종 진단이나 치료 결정은 반드시 전문 의료진이 수행해야 합니다.

특히 심장병 예측 문제에서는 실제 심장병 환자를 정상으로 잘못 예측하는 False Negative가 치명적일 수 있으므로, 모델 평가 시 단순 정확도뿐 아니라 recall, balanced accuracy, confusion matrix를 함께 고려하였습니다.

또한 모델 사용 시 데이터 개인정보 보호, 편향 가능성, 모델 설명 가능성, 지속적인 성능 검증이 함께 고려되어야 합니다.

---

## 20. 한계점 및 개선 방향

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
* FastAPI 기반 API 서빙 구조 추가
* 모니터링 대시보드 구축
* drift 감지 후 Human-in-the-loop 기반 재학습 트리거 구조 설계

---

## 21. AI 도구 사용 공개

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
