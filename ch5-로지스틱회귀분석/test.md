
---

### **로지스틱 회귀(Logistic Regression)란 무엇일까요?**

`ch4`의 타이타닉 예제를 다시 떠올려보세요. 우리는 '생존(1)' 또는 '사망(0)'을 예측하고 싶었지만, 선형 회귀는 1.5나 -0.2 같은 이상한 값을 예측했습니다.

**로지스틱 회귀**는 바로 이런 문제를 해결하기 위해 태어난 **분류 모델**입니다.
핵심 아이디어는 간단합니다.

> "선형 회귀의 예측 결과를 어떤 마법 상자에 넣어, 그 결과가 항상 **0과 1 사이의 확률값**으로 나오게 만들자!"

여기서 '마법 상자'의 역할을 하는 것이 바로 **시그모이드 함수(Sigmoid Function)**입니다.



이 함수는 어떤 값이 들어오든 그 결과를 0과 1 사이의 S자 곡선에 매핑시킵니다.
*   예측값이 아주 크면 -> 1에 가까운 확률 (예: 0.98 -> "98% 확률로 암이다")
*   예측값이 아주 작으면 -> 0에 가까운 확률 (예: 0.05 -> "5% 확률로 암이다")

따라서 로지스틱 회귀는 **"A일 확률이 몇 %인가?"**를 계산하여 분류 문제를 푸는 강력하고 직관적인 도구입니다.

---

### **Ch5-1 코드 종합 설명**

이 코드는 유방암 종양의 여러 세포핵 특성(반지름, 질감 등) 데이터를 사용하여, 해당 종양이 **악성(Malignant)인지 양성(Benign)인지**를 진단하는 이진 분류 모델을 구축하는 과정을 보여줍니다.

1.  **데이터 준비 및 스케일링:** `scikit-learn`에 내장된 유방암 진단 데이터를 불러옵니다. 이 코드의 핵심적인 전처리 과정은 **`StandardScaler`**를 이용한 **특성 스케일링(Feature Scaling)**입니다. 각 특성(반지름, 면적 등)이 서로 다른 단위를 가지므로, 이를 평균 0, 표준편차 1의 정규분포로 변환하여 모델이 더 안정적이고 빠르게 학습할 수 있도록 데이터를 표준화합니다.
2.  **모델 학습:** 데이터를 학습용과 테스트용으로 분할한 뒤, 분류 문제의 대표 주자인 **`LogisticRegression`** 모델을 생성하고 학습(`fit`)시킵니다.
3.  **성능 평가 (분류 지표):** 학습된 모델이 테스트 데이터의 종양 종류를 얼마나 잘 맞추는지 평가합니다. 여기서는 회귀의 MSE, R²가 아닌, 분류 문제에 특화된 **오차 행렬(Confusion Matrix)**을 기반으로 **정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-점수, ROC AUC 점수** 등 다양한 지표를 계산하여 모델의 성능을 다각도로 분석합니다.

---

### **Part 1: 데이터 준비 및 특성 스케일링 (Feature Scaling)**

좋은 모델을 만들기 위해 재료를 손질하는 과정입니다. 특히, 이번 코드에서는 **스케일링**이라는 중요한 전처리 기법이 처음 등장합니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. 데이터 로드
b_cancer = load_breast_cancer()
b_cancer_df = pd.DataFrame(b_cancer.data, columns=b_cancer.feature_names)
b_cancer_df["diagnosis"] = b_cancer.target # 'diagnosis' 컬럼에 정답(0 또는 1) 추가

# %%
# 2. 특성 스케일링
# StandardScaler 객체 생성. 이는 데이터를 표준화하는 '도구'임.
scaler = StandardScaler()

# .fit_transform(b_cancer.data): 데이터(b_cancer.data)를 scaler 도구에 넣어 학습(fit)과 변환(transform)을 동시에 수행.
# - fit: 각 특성(컬럼)의 평균과 표준편차를 계산하여 '기억'함.
# - transform: 기억한 평균과 표준편차를 이용해 각 데이터를 (원래값 - 평균) / 표준편차 공식으로 변환.
b_cancer_scaled = scaler.fit_transform(b_cancer.data)

# 스케일링 전/후 비교
print(f"샘플데이터 정규화 전: {b_cancer.data[0]}")
print(f"샘플데이터 정규화 후: {b_cancer_scaled[0]}")

# %%
# 3. 데이터 분할
# Y(종속 변수)에는 정답 'diagnosis'를 할당
Y = b_cancer_df["diagnosis"]
# X(독립 변수)에는 방금 스케일링을 마친 데이터를 할당
X = b_cancer_scaled

# 훈련용/테스트용 데이터 분할 (70% 훈련, 30% 테스트)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
```

#### **2. 해당 설명**

이 파트의 핵심은 **`StandardScaler`**입니다. 왜 스케일링이 필요할까요?
유방암 데이터의 특성을 보면 `area`(면적)는 값이 수백 단위로 매우 크지만, `smoothness`(매끄러움)는 0.1과 같이 매우 작습니다. 이렇게 값의 범위(스케일)가 크게 다르면, 모델이 `area`와 같이 **숫자가 큰 특성을 더 중요한 것으로 착각**할 수 있습니다.

`StandardScaler`는 모든 특성 데이터를 **"평균이 0, 표준편차가 1인 세상"**으로 옮겨주는 역할을 합니다. 이는 마치 키(cm), 몸무게(kg), 시력(-)처럼 단위가 다른 데이터들을 모두 공평한 출발선에 세우는 것과 같습니다. 이렇게 하면 모델이 각 특성의 순수한 영향력만을 학습할 수 있어 성능이 향상되고 학습 속도도 빨라집니다.

---

### **Part 2: 모델 학습 및 올바른 성능 평가**

분류 모델을 만들고, 이 모델이 얼마나 '똑똑한 의사'인지 평가하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# 1. 모델 생성 및 학습
lr_b_cancer = LogisticRegression()
lr_b_cancer.fit(X_train, Y_train)

# 2. 예측
y_predict = lr_b_cancer.predict(X_test)

# %%
# 3. 분류 성능 평가 지표 계산
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 오차 행렬 (Confusion Matrix) 계산
# Y_test: 실제 정답, y_predict: 모델의 예측
# 결과: [[TN, FP], [FN, TP]] 형태의 2x2 행렬
conf_matrix = confusion_matrix(Y_test, y_predict)
print(f"오차 행렬:\n{conf_matrix}")

# 정확도 (Accuracy)
accuracy = accuracy_score(Y_test, y_predict)
print(f"정확도: {accuracy:.3f}")

# 정밀도 (Precision)
precision = precision_score(Y_test, y_predict)
print(f"정밀도: {precision:.3f}")

# 재현율 (Recall)
recall = recall_score(Y_test, y_predict)
print(f"재현율: {recall:.3f}")

# F1 점수 (F1 Score)
f1 = f1_score(Y_test, y_predict)
print(f"F1 점수: {f1:.3f}")

# ROC AUC 점수
roc_auc = roc_auc_score(Y_test, y_predict)
print(f"ROC AUC: {roc_auc:.3f}")
```

#### **2. 해당 설명**

이 파트가 로지스틱 회귀 학습의 핵심입니다. **오차 행렬(Confusion Matrix)**은 모델의 예측 결과를 4가지 시나리오로 상세하게 보여주는 성적표와 같습니다.

**[유방암 진단 예시로 오차 행렬 해석하기]**
오차 행렬 결과가 `[[60, 3], [4, 104]]` 라고 가정해 봅시다.
*   **TN (True Negative) = 60**: 실제 '양성'(정상)인 환자 60명을 '양성'으로 **올바르게** 진단.
*   **FP (False Positive) = 3**: 실제 '양성'인 환자 3명을 '악성'(암)으로 **잘못** 진단. (멀쩡한데 암이라고 진단)
*   **FN (False Negative) = 4**: 실제 '악성'인 환자 4명을 '양성'으로 **잘못** 진단. (암인데 정상이라고 진단 -> **가장 위험!**)
*   **TP (True Positive) = 104**: 실제 '악성'인 환자 104명을 '악성'으로 **올바르게** 진단.

이 오차 행렬을 기반으로 더 유용한 지표들을 계산합니다.
*   **정확도 (Accuracy)**: `(60+104) / (60+3+4+104) = 164/171 ≈ 0.959`
    *   의미: 전체 환자 중 약 95.9%를 올바르게 진단했다. (매우 높아 보이지만 함정이 있을 수 있음)
*   **정밀도 (Precision)**: `104 / (3+104) ≈ 0.972`
    *   의미: 모델이 "악성입니다"라고 진단한 환자 중, 약 97.2%가 진짜 악성 환자였다. (진단의 신뢰도)
*   **재현율 (Recall)**: `104 / (4+104) ≈ 0.963`
    *   의미: **실제 악성인 환자들 중에서, 모델이 약 96.3%를 놓치지 않고 찾아냈다.**

#### **3. 심화 내용 (정밀도 vs 재현율: 무엇이 더 중요한가?)**

정밀도와 재현율은 종종 **상충 관계(Trade-off)**에 있습니다.
*   **재현율을 높이려면 (FN을 줄이려면)**: 의사가 조금이라도 의심스러우면 "악성일 수 있습니다"라고 진단하는 경향이 생깁니다. 이러면 실제 암 환자를 놓칠 확률은 줄어들지만(재현율 상승), 멀쩡한 사람을 암으로 오진할 확률(FP)이 늘어나 정밀도는 떨어집니다.
*   **정밀도를 높이려면 (FP를 줄이려면)**: 의사가 100% 확실할 때만 "악성입니다"라고 진단합니다. 이러면 오진율은 줄어들지만(정밀도 상승), 애매한 암 환자를 놓칠 확률(FN)이 늘어나 재현율은 떨어집니다.

**유방암 진단에서는 무엇이 더 중요할까요?**
바로 **재현율**입니다. 멀쩡한 사람에게 재검사를 받게 하는 것(FP)보다, 실제 암 환자를 놓치는 것(FN)이 훨씬 치명적이기 때문입니다. 따라서, 이런 의료 진단 모델에서는 재현율을 특히 중요한 지표로 삼습니다. **F1-점수**는 이 둘의 조화 평균으로, 두 지표를 균형 있게 평가하고 싶을 때 사용합니다.