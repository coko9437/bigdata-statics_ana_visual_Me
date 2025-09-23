
---

### **Ch6 코드 종합 설명**

이 코드는 영국의 온라인 소매점 판매 데이터를 사용하여 **고객을 유사한 특성을 가진 그룹으로 나누는** **K-평균 군집 분석**을 수행합니다. 이는 대표적인 **비지도 학습** 방법으로, 정답이 없는 데이터 내에서 숨겨진 패턴이나 구조를 발견하는 것을 목표로 합니다.

1.  **데이터 준비 및 전처리:** 엑셀 파일로 된 원본 데이터를 불러와 분석에 적합하도록 정제합니다. 결측치를 제거하고, 잘못된 데이터(예: 음수 수량)를 삭제하며, 데이터 타입을 변환하는 등 데이터를 깨끗하게 만듭니다.
2.  **RFM 특징 추출:** 고객 행동을 분석하는 데 효과적인 **RFM(Recency, Frequency, Monetary)** 지표를 계산합니다. 고객별로 **마지막 구매 후 경과일(ElapsedDays)**, **구매 빈도(Freq)**, **총 구매 금액(SalesAmount)**을 추출하여 군집 분석에 사용할 핵심 특징(Feature)을 만듭니다.
3.  **데이터 스케일링 및 최적 K 찾기:** K-평균 알고리즘은 변수의 스케일에 민감하므로, 로그 변환과 표준화를 통해 데이터 분포를 조정합니다. 이후 **엘보우 방법(Elbow Method)**을 사용하여 고객을 몇 개의 그룹(K)으로 나누는 것이 가장 적절할지 탐색합니다.
4.  **군집 평가 및 시각화:** 엘보우 방법으로 찾은 K 후보군에 대해 **실루엣 분석(Silhouette Analysis)**을 수행하여 군집이 얼마나 잘 형성되었는지 정량적으로 평가합니다. 실루엣 점수와 군집 분포 산점도를 시각화하여 최적의 K값을 최종 결정합니다.
5.  **결과 해석 및 활용:** 결정된 최적 K값으로 최종 군집 모델을 생성하고, 각 고객이 어떤 그룹에 속하는지 라벨을 부여합니다. 마지막으로 각 그룹의 평균 RFM 특성을 분석하여 'VIP 고객', '이탈 우려 고객' 등과 같은 **고객 세그먼트를 정의하고, 이를 바탕으로 맞춤형 마케팅 전략을 도출**합니다.

---

### **Part 1: 데이터 준비 및 전처리**

모든 분석의 시작은 원본 데이터를 불러와 분석 가능한 형태로 만드는 '정제' 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import pandas as pd
import math

# 엑셀 파일을 읽어와 Pandas DataFrame으로 변환
# ./Online_Retail.xlsx: 현재 폴더에 있는 엑셀 파일
retail_df = pd.read_excel("./Online_Retail.xlsx")

# --- 데이터 기본 정제 ---
# 1. Quantity(주문 수량)가 0보다 큰 데이터만 남김 (음수는 반품 등을 의미하므로 제외)
retail_df = retail_df[retail_df["Quantity"] > 0]
# 2. UnitPrice(제품 단가)가 0보다 큰 데이터만 남김
retail_df = retail_df[retail_df["UnitPrice"] > 0]
# 3. CustomerID(고객 ID)가 비어있지 않은(not null) 데이터만 남김
retail_df = retail_df[retail_df["CustomerID"].notnull()]

# 4. CustomerID의 데이터 타입을 float(실수)에서 int(정수)로 변환
retail_df["CustomerID"] = retail_df["CustomerID"].astype(int)

# --- 중복 데이터 제거 ---
# 데이터프레임에서 완전히 동일한 행(레코드)을 제거
# inplace=True: 변경된 결과를 새로운 변수에 할당하지 않고, 원본 retail_df를 직접 수정
retail_df.drop_duplicates(inplace=True)

# 정제 후 데이터의 기본 정보(행/열 개수, 데이터 타입 등)를 다시 확인
retail_df.info()

# 정제 후 필요한 컬럼들만 사용하여 요약 정보 확인
pd.DataFrame([
    {"Product": len(retail_df["StockCode"].value_counts()),
     "Transaction": len(retail_df["InvoiceNo"].value_counts()),
     "Customer": len(retail_df["CustomerID"].value_counts())}],
    columns=["Product", "Transaction", "Customer"],
    index=["counts"]
)
```

#### **2. 해당 설명**

데이터 분석 프로젝트에서 가장 많은 시간이 소요되는 단계 중 하나가 바로 **데이터 전처리(Data Preprocessing)**입니다. 원본 데이터에는 분석을 방해하는 요소들이 많기 때문입니다. 위 코드에서는 다음과 같은 정제 작업을 수행했습니다.

*   **오류/무의미 데이터 제거:** `Quantity`나 `UnitPrice`가 음수인 경우는 반품이나 시스템 오류일 가능성이 높습니다. `CustomerID`가 없는 데이터는 어떤 고객의 행동인지 특정할 수 없으므로 군집 분석의 목적에 맞지 않습니다. 따라서 이런 데이터들은 분석의 정확도를 위해 제거합니다.
*   **데이터 타입 변환:** `CustomerID`는 고유 식별자이므로 소수점을 가질 이유가 없습니다. `astype(int)`를 통해 명확한 정수형으로 바꿔줍니다.
*   **중복 제거:** 동일한 주문이 실수로 두 번 기록되었을 수 있습니다. `drop_duplicates()`를 통해 이러한 중복 데이터를 제거하여 분석 결과가 왜곡되는 것을 방지합니다.

이러한 과정을 통해 신뢰할 수 있는 분석 결과를 얻기 위한 깨끗한 데이터셋을 준비합니다.

#### **3. 응용 가능한 예제**

**"웹사이트 로그 데이터 분석"**

사용자 행동 로그를 분석할 때, 정상적인 페이지뷰 시간이 아닌 0초 또는 음수 값을 가진 데이터나, 로그인이 되지 않아 사용자 ID가 없는 로그는 분석의 노이즈가 될 수 있습니다. 이런 데이터를 필터링하는 전처리 과정에 위와 같은 기법을 동일하게 적용할 수 있습니다.

#### **4. 추가하고 싶은 내용 (데이터 탐색의 중요성)**

전처리 전 `retail_df.describe()`를 실행하여 각 숫자 컬럼의 기술 통계량(평균, 최소/최대값 등)을 확인하는 것이 좋습니다. 이를 통해 `Quantity`의 최소값이 음수라는 사실을 미리 파악하고, 왜 그런 데이터가 있는지 탐색하여 데이터 정제의 방향을 설정할 수 있습니다.

#### **5. 심화 내용 (대용량 데이터 처리)**

만약 데이터가 수백만 건을 넘어 `pandas`로 한 번에 처리하기 버거울 경우, `Dask`나 `Vaex` 같은 라이브러리를 사용하거나, `pd.read_csv`의 `chunksize` 옵션을 이용해 데이터를 여러 조각으로 나누어 순차적으로 처리하는 방법을 고려할 수 있습니다.

---

### **Part 2: RFM 특징 추출 및 데이터 탐색**

고객을 그룹화하기 위해, 각 고객의 행동 패턴을 나타내는 핵심 지표를 계산하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import datetime
import numpy as np

# 'SalesAmount'(총 구매 금액) 컬럼 생성: 단가 * 수량
retail_df["SalesAmount"] = retail_df["UnitPrice"] * retail_df["Quantity"]

# 고객별로 집계할 기준을 딕셔너리로 정의
aggregations = {
    "InvoiceNo": "count",      # 주문 번호의 개수 -> 구매 빈도(Frequency)
    "SalesAmount": "sum",        # 총 구매 금액의 합 -> 총 구매액(Monetary)
    "InvoiceDate": "max"         # 주문 날짜 중 가장 최근 날짜 -> 마지막 구매일(Recency)
}

# CustomerID를 기준으로 그룹화(groupby)하고, 위에서 정의한 기준으로 집계(agg)
customer_df = retail_df.groupby("CustomerID").agg(aggregations)
# groupby 후 인덱스로 변한 CustomerID를 다시 컬럼으로 변환
customer_df = customer_df.reset_index()

# 분석에 용이하도록 컬럼 이름 변경
customer_df = customer_df.rename(columns={"InvoiceNo": "Freq", "InvoiceDate": "ElapsedDays"})

# 마지막 구매 후 경과일(Recency) 계산
# 기준 날짜(임의로 설정)에서 고객별 마지막 구매일을 빼서 경과 시간(Timedelta)을 계산
customer_df["ElapsedDays"] = datetime.datetime(2011, 12, 10) - customer_df["ElapsedDays"]
# Timedelta 형식에서 .days 속성을 이용해 '일(day)'만 정수로 추출
customer_df["ElapsedDays"] = customer_df["ElapsedDays"].apply(lambda x: x.days + 1)

# --- 데이터 분포 조정 (로그 변환) ---
# 데이터가 한쪽으로 심하게 치우쳐(skewed) 있으므로, 로그 함수를 적용해 분포를 완만하게 만듦
# np.log1p()는 log(1+x)를 계산하여, x가 0일 때 발생하는 오류를 방지
customer_df["Freq_log"] = np.log1p(customer_df["Freq"])
customer_df["SalesAmount_log"] = np.log1p(customer_df["SalesAmount"])
customer_df["ElapsedDays_log"] = np.log1p(customer_df["ElapsedDays"])

# 변환 후 데이터 분포를 박스 플롯으로 시각화하여 확인
fig, ax = plt.subplots()
ax.boxplot([customer_df["Freq_log"], customer_df["SalesAmount_log"], customer_df["ElapsedDays_log"]], sym="bo")
plt.xticks([1, 2, 3], ["Freq_log", "SalesAmount_log", "ElapsedDays_log"])
plt.show()
```

#### **2. 해당 설명**

이 파트에서는 고객 군집 분석의 핵심인 **RFM(Recency, Frequency, Monetary)** 특징을 만듭니다.

*   **Recency (최신성):** `ElapsedDays`. 고객이 마지막으로 언제 구매했는가? (값이 작을수록 최근 고객)
*   **Frequency (빈도):** `Freq`. 고객이 얼마나 자주 구매했는가? (값이 클수록 자주 구매)
*   **Monetary (금액):** `SalesAmount`. 고객이 얼마나 많은 돈을 썼는가? (값이 클수록 VIP)

`groupby()`와 `agg()`는 특정 기준(고객 ID)으로 데이터를 묶어 다양한 집계 연산을 한 번에 수행하는 매우 강력한 기능입니다.

또한, 첫 번째 박스 플롯에서 보았듯 원본 RFM 데이터는 소수의 '큰 손' 고객 때문에 분포가 매우 치우쳐 있습니다. K-평균 같은 거리 기반 알고리즘은 이런 데이터에 잘 동작하지 않으므로, **로그 변환(Log Transformation)**을 통해 데이터의 스케일을 줄이고 분포를 정규분포에 가깝게 만들어 분석 성능을 높입니다. 변환 후 박스 플롯을 보면 중앙값이 박스 중앙에 더 가깝게 위치하고 이상치의 영향이 줄어든 것을 확인할 수 있습니다.

#### **3. 응용 가능한 예제**

**"사용자별 앱 사용 패턴 분석"**

모바일 앱 사용자 데이터를 이용해 `사용자별 접속 빈도(Frequency)`, `총 사용 시간(Monetary)`, `마지막 접속 후 경과일(Recency)`을 계산하여 '헤비 유저', '일반 유저', '휴면 유저' 그룹으로 군집화하고, 각 그룹에 맞는 푸시 알림 전략을 수립할 수 있습니다.

#### **4. 추가하고 싶은 내용 (데이터 시각화)**

박스 플롯 외에 `sns.histplot`이나 `sns.kdeplot`을 사용하여 각 변수의 분포를 히스토그램이나 밀도 곡선으로 시각화하면, 로그 변환 전후의 데이터 분포 변화를 더욱 명확하게 확인할 수 있습니다.

#### **5. 심화 내용 (이상치 처리)**

로그 변환으로도 완전히 해결되지 않는 극단적인 이상치(outlier)는 모델 성능에 악영향을 줄 수 있습니다. IQR(Interquartile Range) 방법을 사용해 상위/하위 1% 데이터를 제거하거나, 다른 값으로 대체(Capping)하는 방법을 추가로 고려하여 모델의 안정성을 높일 수 있습니다.

---

### **Part 3: K-평균 군집 모델링 및 최적 K 탐색**

데이터 준비가 끝나면, 본격적으로 K-평균 알고리즘을 적용하여 고객을 그룹화하는 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 군집 분석에 사용할 특징(로그 변환된 RFM 값)들을 numpy 배열로 추출
X_features = customer_df[["Freq_log", "SalesAmount_log", "ElapsedDays_log"]].values

# --- 데이터 정규화 (표준화) ---
# StandardScaler를 이용해 데이터를 표준 정규분포(평균 0, 표준편차 1)를 따르도록 변환
# fit_transform(): 데이터의 평균과 표준편차를 계산(fit)하고, 이를 이용해 데이터를 변환(transform)
X_features_scaled = StandardScaler().fit_transform(X_features)

# --- 엘보우 방법을 이용한 최적의 K 찾기 ---
# distortions: 각 K값에 대한 군집 내 오차 제곱합(inertia)을 저장할 리스트
distortions = []

# K를 1부터 10까지 변화시키면서 K-Means 모델을 학습
for i in range(1, 11):
    # n_clusters=i: 군집의 개수를 i로 설정하여 모델 생성
    # random_state=0: 결과를 재현할 수 있도록 난수 시드 고정
    kmeans_i = KMeans(n_clusters=i, random_state=0, n_init='auto') # n_init 경고 방지
    # 데이터로 모델을 학습
    kmeans_i.fit(X_features_scaled)
    # 학습된 모델의 inertia_ 값을 리스트에 추가
    # inertia_: 각 데이터 포인트에서 가장 가까운 군집 중심까지의 거리 제곱의 합
    distortions.append(kmeans_i.inertia_)

# K값의 변화에 따른 distortion 값의 변화를 꺾은선 그래프로 시각화
plt.plot(range(1, 11), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortions")
plt.show()
```

#### **2. 해당 설명**

이 파트는 K-평균 군집의 핵심 로직을 담고 있습니다.

1.  **데이터 표준화(Standardization):** `Freq`, `SalesAmount`, `ElapsedDays`는 단위와 스케일이 모두 다릅니다. 이 상태로 거리를 계산하면 스케일이 큰 `SalesAmount`가 군집 결과에 가장 큰 영향을 미치게 됩니다. `StandardScaler`는 모든 변수를 **평균 0, 표준편차 1**인 표준 정규분포로 변환하여, 모든 변수가 공평한 조건에서 거리를 계산하도록 만듭니다.
2.  **엘보우 방법(Elbow Method):** K-평균의 가장 큰 고민은 "고객을 몇 개의 그룹(K)으로 나누어야 하는가?" 입니다. 엘보우 방법은 이 질문에 대한 단서를 제공합니다.
    *   **`inertia_` (왜곡, Distortion):** 군집 내 데이터들이 중심점으로부터 얼마나 떨어져 있는지(얼마나 뭉쳐있는지)를 나타내는 값입니다. 이 값이 작을수록 데이터들이 군집 중심에 잘 모여있다는 의미입니다.
    *   **그래프 해석:** K가 증가할수록 `inertia_`는 당연히 감소합니다 (그룹을 많이 나눌수록 각 그룹의 크기는 작아지므로). 하지만 그래프가 **팔꿈치(Elbow)처럼 급격히 꺾이는 지점**이 있습니다. 이 지점은 K를 더 늘려도 군집의 응집도가 크게 개선되지 않는, 즉 효율이 떨어지기 시작하는 지점입니다. 따라서 이 꺾이는 지점(코드에서는 K=3 또는 4)이 최적의 K가 될 유력한 후보입니다.

#### **3. 응용 가능한 예제**

**"이미지 색상 양자화 (Image Color Quantization)"**

수백만 개의 색상을 가진 이미지를 K개의 대표 색상으로 줄여 이미지 용량을 압축할 때 K-평균을 사용할 수 있습니다. 이때 엘보우 방법을 사용하면, 이미지의 품질을 크게 해치지 않으면서도 효과적으로 색상 수를 줄일 수 있는 최적의 K값을 찾을 수 있습니다.

#### **4. 추가하고 싶은 내용 (random_state의 중요성)**

K-평균은 처음에 군집 중심점을 무작위로 선택하기 때문에, 실행할 때마다 결과가 약간씩 달라질 수 있습니다. `random_state=0`과 같이 시드 값을 고정하면, 코드를 여러 번 실행해도 항상 동일한 초기 중심점을 사용하므로 분석 결과를 일관성 있게 재현할 수 있습니다.

#### **5. 심화 내용 (K-Means++ 초기화)**

`sklearn`의 `KMeans`는 기본적으로 `init='k-means++'` 옵션을 사용합니다. 이는 완전히 무작위로 중심점을 선택하는 대신, 초기 중심점들이 서로 최대한 멀리 떨어지도록 스마트하게 선택하는 방법입니다. 이를 통해 더 빠르고 안정적으로 최적의 군집 결과를 찾는 데 도움이 됩니다.

---

### **Part 4: 군집 평가 및 시각화**

엘보우 방법으로 찾은 K 후보들을 더 정밀하게 평가하여 최종 K를 결정하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import cm

# --- 실루엣 분석 그래프 함수 ---
# n_clusters: 군집 개수, X_features: 분석할 데이터
def silhouetteViz(n_clusters, X_features):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    Y_labels = kmeans.fit_predict(X_features)

    # 모든 데이터 포인트의 실루엣 계수를 계산
    silhouette_values = silhouette_samples(X_features, Y_labels, metric="euclidean")

    # --- 그래프 그리기 ---
    y_ax_lower, y_ax_upper = 0, 0
    # ... (그래프 그리는 코드, 생략) ...
    # 핵심 로직:
    # 1. 각 군집(cluster)에 속하는 데이터들의 실루엣 계수를 가져옴
    # 2. 계수들을 정렬하여 수평 막대그래프(barh)로 그림
    # 3. 전체 실루엣 계수의 평균을 빨간 점선으로 표시

    # 전체 실루엣 계수의 평균값
    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.title(f'Number of Cluster: {n_clusters}\nSilhouette Score: {round(silhouette_avg, 3)}')
    plt.show()


# --- 군집 분포 산점도 함수 ---
# 2차원 평면에 군집 결과를 시각화 (3차원 데이터를 2차원으로 표현)
def clusterScatter(n_clusters, X_features):
    # ... (시각화 코드, 생략) ...
    # 핵심 로직:
    # 1. 각 군집에 속하는 데이터들을 다른 색상의 점(scatter)으로 표시
    # 2. 각 군집의 중심점을 큰 삼각형(^)으로 표시
    plt.show()

# --- K 후보군(3, 4, 5, 6)에 대한 실루엣 분석 및 시각화 실행 ---
silhouetteViz(3, X_features_scaled)
silhouetteViz(4, X_features_scaled)
silhouetteViz(5, X_features_scaled)
silhouetteViz(6, X_features_scaled)

# --- K 후보군에 대한 군집 분포도 시각화 실행 ---
# 원래는 3차원 데이터이므로 PCA 등으로 차원 축소 후 그리는 것이 정확하지만,
# 여기서는 2개 특징(Freq, SalesAmount)만 사용하여 시각적 경향을 확인
clusterScatter(3, X_features_scaled[:, 0:2])
clusterScatter(4, X_features_scaled[:, 0:2])
clusterScatter(5, X_features_scaled[:, 0:2])
clusterScatter(6, X_features_scaled[:, 0:2])
```

#### **2. 해당 설명**

**실루엣 분석(Silhouette Analysis)**은 군집이 얼마나 잘 형성되었는지 평가하는 강력한 도구입니다. 각 데이터 포인트마다 실루엣 계수가 계산되며, 이 값은 -1에서 1 사이의 값을 가집니다.

*   **1에 가까울수록:** 해당 데이터는 자신의 군집에 잘 속해 있고, 다른 군집과는 멀리 떨어져 있음을 의미 (매우 좋음).
*   **0에 가까울수록:** 군집 간의 경계에 위치해 있음을 의미 (애매함).
*   **-1에 가까울수록:** 다른 군집에 더 가까우므로, 잘못 군집화되었을 가능성이 높음 (매우 나쁨).

**실루엣 그래프 해석 방법:**

1.  **전체 평균 실루엣 점수 (빨간 점선):** 이 점수가 높을수록 좋습니다. 일반적으로 0.5 이상이면 괜찮은 군집으로 평가합니다. 코드의 결과에서는 K=3, 4일 때 비교적 점수가 높습니다.
2.  **각 군집의 두께 (막대그래프의 y축 폭):** 각 군집에 속한 데이터의 수를 의미합니다. 군집들의 두께가 비교적 균일한 것이 좋습니다. 특정 군집만 너무 크거나 작으면 좋지 않은 모델일 수 있습니다. K=4일 때 비교적 균일해 보입니다.
3.  **각 군집의 모양:** 막대그래프의 모양이 칼처럼 뾰족하고, 음수 값을 갖는 데이터가 적어야 합니다. K=5, 6으로 갈수록 일부 군집의 실루엣 점수가 평균보다 훨씬 낮아지고 모양이 나빠지는 것을 볼 수 있습니다.

**결론:** 엘보우 방법에서 K=3, 4가 후보였고, 실루엣 분석 결과 K=4일 때 평균 점수가 괜찮고 각 군집의 분포도 비교적 균일하므로, **최적의 K를 4로 결정**하는 것이 합리적입니다.

#### **3. 응용 가능한 예제**

**"뉴스 기사 클러스터링"**

수천 개의 뉴스 기사를 내용에 따라 '정치', '경제', '스포츠' 등의 그룹으로 자동 분류(군집화)한 후, 실루엣 분석을 통해 몇 개의 주제로 나누는 것이 가장 적절한지 평가할 수 있습니다.

#### **4. 추가하고 싶은 내용 (차원 축소와 시각화)**

`clusterScatter` 함수는 3차원(RFM) 데이터를 2차원 평면에 표현하기 위해 2개의 특징만 사용했습니다. 더 정확한 시각화를 위해서는 **주성분 분석(PCA, Principal Component Analysis)**과 같은 차원 축소 기법을 사용하여 3차원 데이터의 정보를 최대한 유지한 채 2차원으로 압축한 뒤, 그 결과를 산점도로 그리는 것이 일반적입니다.

#### **5. 심화 내용 (다른 군집 평가 지표)**

실루엣 점수 외에도 **Calinski-Harabasz Index**, **Davies-Bouldin Index** 등 다양한 군집 평가 지표가 있습니다. 여러 지표를 종합적으로 고려하면 더 신뢰도 높은 최적의 K를 선택할 수 있습니다.

---

### **Part 5: 최종 군집화 및 결과 해석**

결정된 최적의 K로 모델을 만들고, 각 군집이 어떤 특성을 가진 고객 그룹인지 분석하여 비즈니스 인사이트를 도출하는 마지막 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# 최적의 K=4로 설정하여 최종 K-Means 모델을 생성하고 학습
best_cluster = 4
kmeans = KMeans(n_clusters=best_cluster, random_state=0, n_init='auto')
Y_labels = kmeans.fit_predict(X_features_scaled)

# 분석 결과를 원본 customer_df에 'ClusterLabel' 컬럼으로 추가
customer_df['ClusterLabel'] = Y_labels

# --- 결과 해석 ---
# 1. 각 군집(Cluster)에 몇 명의 고객이 있는지 확인
customer_df.groupby('ClusterLabel')['CustomerID'].count()

# 2. 로그 변환 전의 원본 RFM 값과 ClusterLabel만으로 구성된 새로운 데이터프레임 생성
# 각 군집의 특성을 파악하기 위함
customer_cluster_df = customer_df.drop(["Freq_log", "SalesAmount_log", "ElapsedDays_log"], axis=1)

# 3. 'SalesAmountAvg'(1회 평균 구매액) 컬럼 추가
customer_cluster_df["SalesAmountAvg"] = customer_cluster_df["SalesAmount"] / customer_cluster_df["Freq"]

# 4. 각 군집(ClusterLabel)별로 RFM 및 평균 구매액의 '평균' 값을 계산
# 이를 통해 각 군집의 특성을 정의할 수 있음
cluster_summary = customer_cluster_df.drop("CustomerID", axis=1).groupby("ClusterLabel").mean()
print(cluster_summary)

# --- 사용자 코드의 마지막 부분에 대한 설명 ---
# 아래 코드는 '참 값(ground truth)'이 있을 때 사용하는 평가 지표로,
# 이 데이터처럼 정답이 없는 비지도 학습에는 직접 적용할 수 없습니다.
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# ari_score = adjusted_rand_score(true_labels, predicted_labels) # 예시
# nmi_score = normalized_mutual_info_score(true_labels, predicted_labels) # 예시
```

#### **2. 해당 설명**

분석의 최종 목표는 각 군집이 어떤 의미를 갖는지 해석하고 **액션 아이템(Action Item)**을 도출하는 것입니다. `groupby('ClusterLabel').mean()`을 통해 계산된 각 군집의 평균 RFM 값을 보면 그룹의 특성을 파악할 수 있습니다.

**결과 해석 (예시):**
(실제 결과는 실행 시마다 약간 다를 수 있습니다.)

| ClusterLabel | Freq(빈도) | SalesAmount(총액) | ElapsedDays(최신성) | 고객 수 | **고객 세그먼트 정의** | **마케팅 전략** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 낮음 | 낮음 | **높음** | ~891 | **이탈 고객 (Lapsed Customers)** | 재방문을 유도하는 할인 쿠폰 발송, 설문조사 |
| **1** | **높음** | **높음** | **낮음** | ~1207 | **VIP / 충성 고객 (VIP Customers)** | 신제품 우선 체험 기회 제공, 프리미엄 혜택, 감사 메시지 |
| **2** | 보통 | 보통 | 보통 | ~1368 | **일반 고객 (Average Customers)** | 연관 상품 추천, 구매 유도 프로모션 |
| **3** | 낮음 | 낮음 | 낮음 | ~872 | **신규 고객 (New Customers)** | 첫 구매 감사 쿠폰, 앱 사용법 안내, 브랜드 스토리 소개 |

이렇게 고객을 4개의 의미 있는 그룹으로 나누고, 각 그룹의 특성에 맞는 맞춤형 마케팅 전략을 수립하여 비즈니스 성과를 높일 수 있습니다.

#### **3. 응용 가능한 예제**

**"신용카드 사기 탐지(Fraud Detection)"**

정상적인 카드 사용 패턴을 여러 군집으로 나누어 모델링한 후, 어떤 군집에도 속하지 않거나 군집 중심에서 매우 멀리 떨어진 새로운 거래 데이터가 발생하면 이를 '의심스러운 거래(이상치)'로 탐지하여 조사를 요청하는 시스템을 만들 수 있습니다.

#### **4. 추가하고 싶은 내용 (ARI, NMI 평가 지표)**

사용자께서 마지막에 추가한 `adjusted_rand_score`(ARI)와 `normalized_mutual_info_score`(NMI)는 군집 분석의 성능을 평가하는 훌륭한 지표입니다. 하지만 이 지표들은 **정답 라벨이 있을 때만 사용 가능**합니다. 예를 들어, 붓꽃(Iris) 데이터처럼 품종(setosa, versicolor 등)이라는 정답이 있는 데이터를 군집화한 후, "알고리즘이 만든 군집"과 "실제 품종"이 얼마나 일치하는지를 측정할 때 사용합니다. 이번 온라인 판매 데이터처럼 정답이 없는 경우에는 사용할 수 없으며, 그래서 실루엣 점수 같은 내부 평가 지표를 사용하는 것입니다.

#### **5. 심화 내용 (다른 군집 알고리즘)**

K-평균은 원형의 군집을 잘 찾아내지만, 길쭉하거나 밀도가 다른 군집은 잘 찾아내지 못하는 단점이 있습니다. 이런 경우에는 **DBSCAN**(밀도 기반 군집), **GMM**(Gaussian Mixture Model, 분포 기반 군집), **계층적 군집(Hierarchical Clustering)** 등 다른 군집 알고리즘을 사용하면 더 좋은 결과를 얻을 수도 있습니다. 데이터의 특성에 맞는 적절한 알고리즘을 선택하는 것이 중요합니다.