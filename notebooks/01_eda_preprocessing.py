import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/heart.csv"

# 1. 데이터 불러오기
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("1. 데이터 기본 확인")
print("=" * 50)

print("\n[데이터 크기]")
print(df.shape)

print("\n[앞 5행]")
print(df.head())

print("\n[컬럼 목록]")
print(df.columns.tolist())

print("\n[데이터 정보]")
print(df.info())

print("\n[기초 통계량]")
print(df.describe())


print("\n" + "=" * 50)
print("2. 타깃 클래스 분포 확인")
print("=" * 50)

print("\n[target 개수]")
print(df["target"].value_counts())

print("\n[target 비율]")
print(df["target"].value_counts(normalize=True))


print("\n" + "=" * 50)
print("3. 결측값 확인")
print("=" * 50)

print("\n[컬럼별 결측값]")
print(df.isnull().sum())

print("\n[전체 결측값 개수]")
print(df.isnull().sum().sum())


print("\n" + "=" * 50)
print("4. 중복값 확인")
print("=" * 50)

print("\n[중복 행 개수]")
print(df.duplicated().sum())


print("\n" + "=" * 50)
print("5. 연속형 변수 이상치 확인")
print("=" * 50)

continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

print("\n[연속형 변수 기초 통계]")
print(df[continuous_cols].describe())

# Boxplot 저장
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[continuous_cols])
plt.title("Boxplot of Continuous Features")
plt.tight_layout()
plt.savefig("reports/continuous_features_boxplot.png")
plt.show()


print("\n" + "=" * 50)
print("6. target 분포 시각화")
print("=" * 50)

plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Target Class Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/target_distribution.png")
plt.show()


print("\nEDA 완료")