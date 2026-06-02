# CardioCare

## 1. 프로젝트 개요

CardioCare는 임상 데이터를 기반으로 심장병 발병 가능성을 예측하는 머신러닝 프로젝트입니다.

이 프로젝트의 목표는 UCI Heart Disease Dataset을 활용하여 심장병 위험 여부를 예측하는 분류 모델을 만들고, 데이터 전처리부터 모델 학습, 평가, 테스트, 패키징, 모니터링까지 이어지는 End-to-End 머신러닝 시스템을 구축하는 것입니다.

CardioCare는 의료진의 의사결정을 보조하기 위한 시스템이며, 단독으로 진단을 내리는 시스템이 아닙니다.

> Inform, not decide.
> CardioCare는 판단을 대신하는 것이 아니라, 판단에 참고할 수 있는 정보를 제공하는 것을 목표로 합니다.

---

## 2. 사용 데이터셋

본 프로젝트에서는 UCI Heart Disease Dataset을 사용합니다.

* 데이터셋 이름: UCI Heart Disease Dataset
* 사용 목적: 환자의 임상 데이터를 기반으로 심장병 여부 예측
* 문제 유형: 이진 분류(Binary Classification)

데이터는 `data/heart.csv` 경로에 저장하여 사용합니다.

타깃 값이 여러 클래스로 구성되어 있는 경우, 다음과 같이 이진화하여 사용합니다.

기존 target 값 | 변환 후 의미
 0  | 정상
 1 이상  | 심장병 가능성 있음 

최종 target

| target | 의미
 0       | No Heart Disease      
 1       | Heart Disease Present 

---

## 3. 프로젝트 구조

CardioCare/
├── data/
│   └── heart.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── models/
├── reports/
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore

각 폴더와 파일의 역할은 다음과 같습니다.

| 경로                       | 설명                           |
| ------------------------ | ---------------------------- |
| `data/`                  | 데이터셋 저장 폴더                   |
| `notebooks/`             | EDA 및 전처리 분석 노트북             |
| `src/preprocessing.py`   | 데이터 로딩 및 전처리 파이프라인 코드        |
| `src/train.py`           | 모델 학습, 평가, MLflow 기록 코드      |
| `src/inference.py`       | 저장된 모델을 이용한 예측 코드            |
| `src/monitor.py`         | 추론 로깅 및 데이터 드리프트 탐지 코드       |
| `tests/test_pipeline.py` | 머신러닝 파이프라인 단위 테스트            |
| `models/`                | 학습된 모델 저장 폴더                 |
| `reports/`               | 보고서용 그림, 표, 결과 저장 폴더         |
| `requirements.txt`       | 프로젝트 실행에 필요한 Python 라이브러리 목록 |
| `Dockerfile`             | Docker 이미지 빌드 설정 파일          |
| `README.md`              | 프로젝트 설명 및 실행 방법 문서           |

---

## 4. 개발 환경

본 프로젝트는 다음 환경을 기준으로 작성되었습니다.

* Python 3.10 이상
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* scipy
* mlflow
* joblib
* jupyter

자세한 라이브러리 버전은 `requirements.txt` 파일에 명시합니다.

---

## 5. 설치 방법

프로젝트를 실행하기 위해 먼저 가상환경을 생성합니다.

```bash
python -m venv .venv
```

Windows PowerShell 기준으로 가상환경을 실행합니다.

```bash
.venv\Scripts\activate
```

필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

---

## 6. 데이터 준비

데이터 파일은 다음 경로에 위치해야 합니다.

```text
data/heart.csv
```

데이터 파일이 없는 경우, UCI Heart Disease Dataset 또는 동일한 구조의 Kaggle Heart Disease CSV 파일을 다운로드한 뒤 `heart.csv`라는 이름으로 저장합니다.

예상되는 주요 컬럼은 다음과 같습니다.

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

---

## 7. 실행 방법

### 7.1 EDA 및 전처리 확인

EDA와 전처리 분석은 다음 노트북에서 확인할 수 있습니다.

```text
notebooks/01_eda_preprocessing.ipynb
```

이 노트북에서는 다음 내용을 확인합니다.

* 데이터 기본 구조 확인
* `head()`, `info()`, `describe()` 결과 확인
* 타깃 클래스 분포 확인
* 결측값 확인
* 이상치 확인
* 전처리 방향 정리

---

### 7.2 모델 학습 실행

모델 학습은 다음 명령어로 실행합니다.

```bash
python src/train.py
```

학습 과정에서는 다음 모델들을 비교합니다.

* Logistic Regression
* Support Vector Classifier
* Random Forest Classifier

평가 지표는 다음을 사용합니다.

* Balanced Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

의료 예측 문제에서는 실제 심장병 환자를 정상으로 잘못 예측하는 False Negative가 중요하므로, 최종 모델 선택 시 Recall과 Balanced Accuracy를 함께 고려합니다.

---

### 7.3 MLflow 실행

모델 학습 실험 결과는 MLflow를 통해 기록합니다.

MLflow UI는 다음 명령어로 실행할 수 있습니다.

```bash
mlflow ui
```

실행 후 브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:5000
```

MLflow에는 다음 정보를 기록합니다.

* 모델 이름
* 하이퍼파라미터
* Balanced Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* 학습된 모델 아티팩트

---

### 7.4 테스트 실행

단위 테스트는 다음 명령어로 실행합니다.

```bash
python -m unittest
```

테스트 항목은 다음과 같습니다.

1. 예측 결과의 shape가 입력 데이터와 일치하는지 확인
2. 예측 확률이 0과 1 사이에 있는지 확인
3. 임상적으로 정해진 입력 범위를 벗어나는 값이 있는지 확인
4. 고정된 random seed에서 동일 입력에 대해 동일한 결과가 나오는지 확인

---

### 7.5 모니터링 및 데이터 드리프트 탐지

모니터링 코드는 다음 명령어로 실행합니다.

```bash
python src/monitor.py
```

모니터링 단계에서는 다음 내용을 수행합니다.

* 추론 로그 저장
* 모델 버전, 입력 shape, 예측값 기록
* 테스트 데이터의 일부 연속형 특성을 인위적으로 이동
* KS 검정을 이용한 데이터 드리프트 탐지
* 원본 테스트셋과 드리프트 테스트셋의 성능 비교

---

## 8. Docker 실행 방법

Docker 이미지는 다음 명령어로 빌드합니다.

```bash
docker build -t cardiocare:1.0 .
```

빌드한 이미지는 다음 명령어로 실행합니다.

```bash
docker run cardiocare:1.0
```

Docker를 사용하는 이유는 실행 환경 차이로 인한 오류를 줄이고, 다른 컴퓨터에서도 동일한 방식으로 프로젝트를 재현할 수 있도록 하기 위함입니다.

---

## 9. CI Workflow

본 프로젝트는 GitHub Actions를 이용하여 기본 CI 워크플로를 구성합니다.

CI에서는 push가 발생할 때마다 다음 작업을 자동으로 수행합니다.

```bash
python -m unittest
```

이를 통해 코드 변경 후에도 기본 테스트가 통과하는지 확인합니다.

---

## 10. 윤리적 고려 사항

CardioCare는 심장병 예측을 위한 보조 도구입니다.

이 모델의 예측 결과는 의료진의 판단을 보조하기 위한 참고 자료로만 사용되어야 하며, 환자에 대한 최종 진단이나 치료 결정은 반드시 전문 의료진이 수행해야 합니다.

특히 심장병 예측 문제에서는 실제 심장병 환자를 정상으로 잘못 예측하는 False Negative가 치명적일 수 있습니다. 따라서 모델 평가 시 단순 정확도뿐 아니라 Recall, Balanced Accuracy, Confusion Matrix를 함께 고려합니다.

---

## 11. 한계점

본 프로젝트는 공개 데이터셋을 기반으로 한 학습용 프로젝트이므로 실제 의료 환경에 바로 적용하기에는 한계가 있습니다.

주요 한계는 다음과 같습니다.

* 데이터셋의 크기가 제한적임
* 실제 병원 환경의 다양한 환자군을 모두 반영하지 못할 수 있음
* 데이터 수집 시점과 실제 적용 시점의 분포 차이가 발생할 수 있음
* 모델 예측 결과에 대한 의학적 검증이 별도로 필요함

---

## 12. AI 도구 사용 공개

본 프로젝트를 진행하는 과정에서 AI 도구를 일부 활용할 수 있습니다.

AI 도구는 다음 목적으로만 사용합니다.

* 코드 구조 설계 보조
* 오류 해결 및 디버깅 방향 확인
* README 및 보고서 초안 작성 보조
* 머신러닝 개념 이해 보조

단, 최종 제출 코드, 실험 결과, 보고서 내용은 작성자가 직접 검토하고 수정하며, 제출물에 대한 책임은 작성자 본인에게 있습니다.

---

## 13. 추후 개선 방향

추가 시간이 주어진다면 다음 내용을 개선할 수 있습니다.

* 더 다양한 모델과 하이퍼파라미터 탐색
* 데이터 불균형 처리 기법 적용
* 모델 해석을 위한 feature importance 분석 강화
* SHAP 등을 활용한 예측 설명 가능성 개선
* 실제 서비스 환경을 가정한 API 서빙 구조 추가
* 모델 재학습 자동화 전략 구체화

---

## 14. 제출물

최종 제출물은 다음과 같습니다.

* GitHub Repository 링크
* `report.pdf`
* 실행 가능한 코드
* MLflow 실험 결과
* Dockerfile
* CI Workflow
* README.md
