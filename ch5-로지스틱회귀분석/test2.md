
---

### **결정 트리(Decision Tree)란 무엇일까요?**

결정 트리는 이름 그대로 **'결정을 내리는 나무'** 모양의 모델입니다. 스무고개 게임을 생각하면 아주 쉽습니다.

> **"제가 생각하는 동물이 무엇인지 맞춰보세요!"**

*   **질문 1: "날개가 있나요?"** -> 예 / 아니오
*   **(예) 질문 2: "스스로 날 수 있나요?"** -> 예 / 아니오
*   **(예) 답: "참새!"**
*   **(아니오) 답: "펭귄!"**
*   **(아니오) 질문 3: "털이 있나요?"** -> 예 / 아니오
*   **(예) 답: "다람쥐!"**
*   **(아니오) 답: "개구리!"**

이 스무고개 과정 자체가 바로 결정 트리입니다.
*   **규칙 노드 (Rule Node)**: "날개가 있나요?"와 같이 데이터를 나누는 **질문(조건)**.
*   **리프 노드 (Leaf Node)**: "참새", "펭귄"과 같이 최종적으로 내려지는 **결정(답)**.

결정 트리 모델은 데이터를 가장 잘 나눌 수 있는 '최적의 질문'을 스스로 찾아내어 이 나무 구조를 만들어 나갑니다. "어떤 질문을 먼저 해야 정답을 빨리 맞힐 수 있을까?"를 고민하는 것과 같습니다.

**어떻게 '최적의 질문'을 찾을까요?**
이때 **정보 이득(Information Gain)**이나 **지니 계수(Gini Index)** 같은 통계적 척도를 사용합니다. 두 개념 모두 목표는 하나입니다.

> **"이 질문으로 데이터를 나눴을 때, 각 그룹이 얼마나 '순수'해지는가?"**

'순수한 그룹'이란, 한 그룹 안에 같은 종류의 데이터만 모여있는 상태를 말합니다. 예를 들어 "날개가 있나요?"라는 질문에 '예'로 답한 그룹에 참새와 펭귄만 있고, '아니오' 그룹에 다람쥐와 개구리만 있다면, 이 질문은 데이터를 잘 나눈 '좋은 질문'이 됩니다. 결정 트리는 이런 좋은 질문들을 계속 찾아가며 나무를 키워나갑니다.

---

### **Ch5-2 코드 종합 설명**

이 코드는 **결정 트리 분류기**를 사용하여 스마트폰 센서 데이터로부터 사람의 **행동(걷기, 앉기, 눕기 등 6가지)**을 예측하는 다중 분류(Multi-class Classification) 문제를 해결합니다.

1.  **데이터 준비:** 여러 텍스트 파일로 나뉘어 있는 센서 데이터를 불러와, `pandas`를 이용해 학습용(X_train, Y_train)과 테스트용(X_test, Y_test) 데이터프레임으로 재구성합니다. `features.txt` 파일의 정보를 활용하여 561개의 복잡한 센서 데이터 컬럼에 이름을 부여합니다.
2.  **기본 모델 학습 및 평가:** `DecisionTreeClassifier`를 기본 설정으로 생성하여 학습시키고, 테스트 데이터에 대한 **정확도(Accuracy)**를 측정하여 모델의 초기 성능을 확인합니다.
3.  **하이퍼파라미터 튜닝 (GridSearchCV):** 결정 트리의 성능에 큰 영향을 미치는 **`max_depth`(트리의 최대 깊이)**, **`min_samples_split`(노드를 나눌 최소 샘플 수)** 등의 **하이퍼파라미터**를 최적화합니다. `GridSearchCV`를 사용하여 여러 조합을 자동으로 테스트하고, **교차 검증(Cross-Validation)**을 통해 가장 성능이 좋은 최적의 파라미터 조합을 찾아냅니다.
4.  **최적 모델 평가 및 해석:** 찾아낸 최적의 파라미터로 최종 모델을 다시 학습시키고 성능을 평가합니다. 또한, 모델의 `feature_importances_` 속성을 분석하여 561개의 센서 신호 중 어떤 신호가 사람의 행동을 예측하는 데 가장 **중요한 역할**을 했는지 파악하고 시각화합니다.
5.  **결정 트리 시각화:** `Graphviz` 라이브러리를 이용해, 학습된 결정 트리 모델이 어떤 '질문(규칙)'들로 구성되어 있는지 직접 눈으로 확인할 수 있는 이미지 파일로 생성합니다.

---

### **Part 1: 복잡한 데이터 준비 및 모델링**

여러 파일로 분산된 데이터를 불러와 하나의 분석 가능한 형태로 만드는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 피처(컬럼) 이름 불러오기
# features.txt 파일에서 561개 피처의 이름을 읽어 리스트로 저장
feature_name_df = pd.read_csv("./UCI HAR Dataset/features.txt", sep="\s+", header=None, names=["index", "feature_name"], engine="python")
feature_name = feature_name_df.iloc[:, 1].values.tolist()

# 2. 학습용/테스트용 데이터 불러오기
# X_train.txt, y_train.txt, X_test.txt, y_test.txt 파일들을 각각 DataFrame으로 로드
X_train = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", sep="\s+", header=None, engine="python")
Y_train = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", sep="\s+", header=None, names=["action"], engine="python")
X_test = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", sep="\s+", header=None, engine="python")
Y_test = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", sep="\s+", header=None, names=["action"], engine="python")

# 3. 데이터프레임에 컬럼 이름 부여
# 불러온 X_train, X_test 데이터프레임의 컬럼에 feature_name 리스트를 적용
X_train.columns = feature_name
X_test.columns = feature_name

# %%
# 4. 기본 결정 트리 모델 생성 및 학습
# DecisionTreeClassifier 객체를 기본 설정으로 생성
dt_HAR = DecisionTreeClassifier()
# 학습용 데이터로 모델을 훈련시킴
dt_HAR.fit(X_train, Y_train)

# 5. 예측 및 기본 정확도 평가
Y_predict = dt_HAR.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predict)
print(f"기본 정확도: {accuracy:.4f}") # 약 86%의 준수한 초기 성능을 보임
```

#### **2. 해당 설명**

이 파트는 실제 데이터 분석 프로젝트가 어떻게 시작되는지를 잘 보여줍니다. 데이터는 항상 깔끔한 CSV 파일 하나로 주어지지 않습니다. 이처럼 여러 파일에 걸쳐 데이터, 컬럼명, 정답 등이 나뉘어 있는 경우가 흔하며, `pandas`를 이용해 이를 능숙하게 조합하는 능력이 중요합니다. `sep="\s+"` 옵션은 데이터가 일정하지 않은 공백으로 구분되어 있을 때 유용하게 사용됩니다.

우선 기본 모델의 정확도를 측정하는 것은, 앞으로 진행할 **튜닝(Tuning)** 과정이 얼마나 효과가 있었는지 비교하기 위한 **기준점(Baseline)**을 설정하는 중요한 단계입니다.

---

### **Part 2: 최적의 모델 찾기 (GridSearchCV)**

모델의 성능을 극한으로 끌어올리기 위한 '튜닝' 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from sklearn.model_selection import GridSearchCV

# 1. 튜닝할 하이퍼파라미터 후보군 정의
# "max_depth"라는 파라미터를 6, 8, 10, 12, 16, 20, 24로 바꿔가며 테스트하겠다는 의미
params = {
    "max_depth": [6, 8, 10, 12, 16, 20, 24]
}

# 2. GridSearchCV 객체 생성
# - dt_HAR: 튜닝할 기본 모델
# - param_grid=params: 테스트할 하이퍼파라미터 조합
# - scoring="accuracy": 성능 평가 기준을 '정확도'로 설정
# - cv=5: 교차 검증(Cross-Validation)을 5번 수행. (데이터를 5조각으로 나눠 4개로 학습, 1개로 검증하는 과정을 5번 반복)
grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring="accuracy", cv=5, return_train_score=True)

# 3. 튜닝 실행 (학습)
# .fit()을 호출하면, params에 정의된 모든 조합(여기서는 7개)에 대해 교차 검증을 자동으로 수행함
grid_cv.fit(X_train, Y_train)

# 4. 튜닝 결과 확인
print(f"최고 평균 정확도: {grid_cv.best_score_:.4f}")
print(f"최적의 하이퍼파라미터: {grid_cv.best_params_}")

# 5. 최적의 모델로 최종 평가
# grid_cv는 가장 좋았던 파라미터로 학습된 모델을 best_estimator_ 속성에 저장하고 있음
best_dt_HAR = grid_cv.best_estimator_

# 최적 모델을 이용해 테스트 데이터로 예측
best_Y_predict = best_dt_HAR.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)
print(f"베스트 결정 트리 예측 정확도: {best_accuracy:.4f}")
```

#### **2. 해당 설명**

이 파트의 핵심은 **`GridSearchCV`**입니다. 결정 트리를 그냥 만들면, 나무가 너무 깊고 복잡하게 자라 훈련 데이터는 100% 맞추지만 새로운 데이터는 전혀 못 맞추는 **과적합(Overfitting)**에 빠지기 쉽습니다. 이를 방지하기 위해 `max_depth` (나무의 최대 깊이) 같은 제약을 두어야 하는데, 최적의 깊이가 얼마인지는 아무도 모릅니다.

`GridSearchCV`는 우리가 지정한 파라미터 후보들을 **모두 자동으로 테스트**하여 최적의 값을 찾아주는 매우 편리한 도구입니다. 특히 **교차 검증(cv=5)**을 통해 데이터를 여러 번 나눠 평가하므로, 우연히 얻어진 성능이 아닌 일반화되고 안정적인 성능을 기준으로 최적의 파라미터를 선택해줍니다. 이 과정을 통해 기본 모델의 정확도(약 86%)보다 더 높은, 안정적인 성능(약 85%대)을 가진 모델을 찾을 수 있었습니다. (때로는 튜닝 후 점수가 약간 떨어질 수도 있는데, 이는 과적합이 방지된 더 안정적인 모델이라는 의미일 수 있습니다.)

---

### **Part 3: 모델 해석 및 시각화**

모델이 무엇을 배웠는지 들여다보고, 그 구조를 직접 눈으로 확인하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# 1. 특성 중요도 분석
# .feature_importances_: 각 피처(컬럼)가 모델의 예측에 얼마나 중요한 역할을 했는지 0~1 사이의 값으로 나타냄.
feature_importances_values = best_dt_HAR.feature_importances_

# 중요도를 보기 쉽게 pandas Series로 변환
feature_importances_values_s = pd.Series(feature_importances_values, index=X_train.columns)

# 중요도가 높은 상위 10개 피처를 추출
feature_top10 = feature_importances_values_s.sort_values(ascending=False)[:10]

# %%
# 2. 특성 중요도 시각화
plt.figure(figsize=(10, 5))
plt.title("Feature Top 10")
# seaborn의 barplot으로 중요도를 막대그래프로 표현
sns.barplot(x=feature_top10, y=feature_top10.index)
plt.show()

# %%
# 3. 결정 트리 구조 시각화
from sklearn.tree import export_graphviz
import graphviz

# 학습된 결정 트리 모델의 구조를 'tree.dot'이라는 텍스트 파일로 내보냄
export_graphviz(best_dt_HAR, out_file="tree.dot", class_names=label_name, feature_names=feature_name, impurity=True, filled=True)

# .dot 파일을 읽어 graphviz 객체로 변환
with open("tree.dot") as f:
    dot_graph = f.read()
src = graphviz.Source(dot_graph)

# 객체를 이미지 파일(png)로 렌더링하고 저장
src.render('tree_image', format='png', view=False)
```

#### **2. 해당 설명**

**특성 중요도(`feature_importances_`)** 분석은 결정 트리의 가장 큰 장점 중 하나입니다. 561개의 복잡한 센서 신호 중, 모델이 사람의 행동을 구분하기 위해 **"어떤 신호를 주로 사용했는가"**를 알려줍니다. 막대그래프를 보면, 특정 각도나 가속도 신호 몇 개가 예측에 결정적인 역할을 했음을 알 수 있습니다. 이는 단순히 예측만 잘하는 것을 넘어, **"왜" 그런 예측을 했는지에 대한 설명(Explainability)**을 제공하여 분석에 깊이를 더해줍니다.

**`Graphviz`를 이용한 시각화**는 결정 트리의 '두뇌'를 직접 들여다보는 것과 같습니다. 생성된 이미지 파일(`tree_image.png`)을 열어보면, "tGravityAcc-min()-X <= -0.925인가?"와 같은 실제 '질문'들로 구성된 나무 구조를 볼 수 있습니다. 각 노드의 색깔은 어떤 행동(걷기, 앉기 등)이 다수를 차지하는지를, `gini` 값은 해당 노드의 데이터가 얼마나 순수한지를 나타냅니다. 이를 통해 모델의 의사결정 과정을 투명하게 이해할 수 있습니다.