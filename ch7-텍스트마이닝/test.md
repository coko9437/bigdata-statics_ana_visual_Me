
---

### **Ch7-1 코드 종합 설명**

이 코드는 텍스트 마이닝의 두 가지 핵심 기법인 **감성 분석(Sentiment Analysis)**과 **토픽 모델링(Topic Modeling)**을 수행하는 파이프라인을 구축합니다. 전체적인 흐름은 다음과 같습니다.

1.  **감성 분석 모델 구축 (네이버 영화 리뷰):**
    *   **데이터 준비:** 네이버 영화 리뷰 데이터(`ratings_train.txt`)를 불러와 긍정(1) 또는 부정(0) 라벨이 달린 텍스트를 정제합니다. 한글 이외의 불필요한 문자를 제거하는 전처리 과정을 거칩니다.
    *   **특징 벡터화:** 컴퓨터가 텍스트를 이해할 수 있도록, 한국어 형태소 분석기(`Konlpy Okt`)를 사용해 문장을 단어로 나누고, 이를 **TF-IDF(Term Frequency-Inverse Document Frequency)** 방식으로 숫자 벡터로 변환합니다. 이 과정은 "어떤 단어가 이 문장의 감정을 나타내는 데 중요한가?"를 수치화하는 단계입니다.
    *   **모델 학습 및 최적화:** 벡터화된 데이터를 **로지스틱 회귀(Logistic Regression)** 모델에 학습시켜 긍정/부정을 분류하는 규칙을 찾게 합니다. `GridSearchCV`를 이용해 교차 검증을 수행하며 모델의 성능을 최대로 끌어올리는 최적의 하이퍼파라미터를 찾습니다.
    *   **모델 평가 및 활용:** 학습된 모델을 평가용 데이터(`ratings_test.txt`)로 성능(정확도)을 검증하고, 사용자가 직접 입력한 문장의 감성을 실시간으로 예측해봅니다.

2.  **새로운 데이터에 감성 분석 적용 (코로나 뉴스):**
    *   앞서 구축한 영화 리뷰 감성 분석 모델을 전혀 다른 도메인인 **코로나 관련 뉴스 데이터**에 적용합니다. 뉴스 기사의 제목과 본문이 긍정적인지 부정적인지 예측하여 라벨을 부여합니다.
    *   긍정/부정으로 분류된 뉴스 그룹에서 각각 어떤 단어들이 가장 빈번하게 등장하는지 분석하고, 이를 바 차트로 시각화하여 각 감성 그룹의 주요 키워드를 파악합니다.

3.  **토픽 모델링 (코로나 뉴스):**
    *   코로나 뉴스 데이터 전체를 대상으로 **LDA(Latent Dirichlet Allocation)** 알고리즘을 사용해 기사들에 잠재된 핵심 주제(토픽)들을 자동으로 추출합니다. 이는 "사람들이 코로나와 관련하여 주로 어떤 주제에 대해 이야기하고 있는가?"를 발견하는 과정입니다.
    *   `pyLDAvis` 라이브러리를 사용해 추출된 토픽 간의 관계와 각 토픽을 구성하는 주요 단어들을 인터랙티브하게 시각화하여 분석 결과를 직관적으로 이해합니다.

---

### **Part 1: 감성 분석 - 데이터 준비 및 전처리 (네이버 영화 리뷰)**

모델 학습을 위해 잘 정제된 훈련 데이터를 만드는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import pandas as pd
import re

# 훈련용 데이터 로드
# sep='\t': 데이터가 탭(tab)으로 구분되어 있음을 명시
nsmc_train_df = pd.read_csv("./ratings_train.txt", encoding="utf-8", sep="\t", engine="python")

# --- 데이터 정제 ---
# 1. 'document' 컬럼에 비어있는 값(null)이 있는 행을 제거
nsmc_train_df = nsmc_train_df[nsmc_train_df["document"].notnull()]

# 2. 'document' 컬럼에 정규표현식(regex)을 적용하여 한글과 공백을 제외한 모든 문자를 제거
# re.sub(pattern, replace, string): string에서 pattern에 해당하는 부분을 replace로 바꿈
# [^ ㄱ - | 가-힣]+ : 'ㄱ'부터 '|', '가'부터 '힣'까지의 문자를 제외한(^), 모든 문자가 1개 이상(+) 반복되는 패턴
# 즉, 한글 자모음, 완성형 한글, 특정 기호, 공백을 제외한 모든 것을 공백(" ")으로 치환
nsmc_train_df["document"] = nsmc_train_df["document"].apply(lambda x: re.sub(r'[^ ㄱ - | 가-힣]+', " ", x))

# 평가용 데이터(test set)에도 동일한 전처리 과정을 적용
nsmc_test_df = pd.read_csv("./ratings_test.txt", encoding="utf-8", sep="\t", engine="python")
nsmc_test_df = nsmc_test_df[nsmc_test_df["document"].notnull()]
nsmc_test_df["document"] = nsmc_test_df["document"].apply(lambda x: re.sub(r'[^ ㄱ - | 가-힣]+', " ", x))
```

#### **2. 해당 설명**

**텍스트 전처리**는 텍스트 마이닝의 성패를 좌우하는 가장 중요한 단계입니다. 컴퓨터는 텍스트 자체를 이해하지 못하므로, 분석에 방해가 되는 불필요한 정보(노이즈)를 제거하고 의미 있는 텍스트만 남겨야 합니다.

*   **결측치 제거:** 내용이 없는 리뷰는 분석할 수 없으므로 `notnull()`을 통해 제거합니다.
*   **정규표현식을 이용한 한글 추출:** 영화 리뷰에는 특수문자, 이모티콘, 영어, 숫자 등 감성 분석에 직접적인 도움이 되지 않는 요소들이 많습니다. `re.sub`과 한글 정규표현식을 사용해 이러한 노이즈를 효과적으로 제거하고 오직 **한글 텍스트**만 남겨 모델이 핵심 단어에 집중하도록 합니다.

훈련 데이터와 평가용 데이터에 **반드시 동일한 전처리 규칙**을 적용해야, 모델이 일관된 기준으로 예측을 수행할 수 있습니다.

#### **3. 응용 가능한 예제**

**"상품평 데이터에서 욕설 및 비속어 필터링"**

정규표현식을 사용하여 데이터베이스에 저장된 상품평에서 욕설이나 비속어 패턴을 찾아내어 `***` 등으로 마스킹(masking)하는 전처리 로직을 구현할 수 있습니다.

#### **4. 추가하고 싶은 내용 (불용어 처리)**

실제 분석에서는 '은', '는', '이', '가', '있다' 등 문법적으로는 필요하지만 의미 분석에는 큰 도움이 되지 않는 **불용어(Stopwords)**를 제거하는 과정이 추가되는 경우가 많습니다. 이는 모델이 더 중요한 의미어에 집중하게 하여 성능을 향상시킬 수 있습니다.

#### **5. 심화 내용 (형태소 분석 기반 정제)**

단순히 한글만 남기는 것을 넘어, `Okt.normalize()` 같은 함수를 사용해 "ㅋㅋㅋㅋㅋ"나 "ㅎㅎㅎㅎ"처럼 반복되는 자음을 "ㅋㅋ", "ㅎㅎ" 등으로 정규화하거나, 오타를 교정하는 등 더 정교한 전처리 기법을 적용하여 데이터의 품질을 높일 수 있습니다.

---

### **Part 2: 특징 벡터화 (TF-IDF) 및 모델 저장**

정제된 텍스트를 머신러닝 모델이 학습할 수 있는 숫자 형태의 데이터(벡터)로 변환하는 핵심 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# --- 1. 형태소 분석기 정의 ---
# Okt (Open Korean Text) 객체 생성. 트위터에서 개발한 한국어 형태소 분석기
okt = Okt()
# 문장을 형태소 단위로 쪼개는 함수 정의
def okt_tokenizer(text):
    tokens = okt.morphs(text)  # morphs(): 텍스트를 형태소 배열로 반환
    return tokens

# --- 2. TF-IDF 벡터화 모델 생성 ---
# TfidfVectorizer 객체 생성
tfidf = TfidfVectorizer(
    tokenizer=okt_tokenizer,      # 위에서 정의한 Okt 형태소 분석기를 사용
    ngram_range=(1, 2),         # 단어를 1개(unigram) 또는 2개(bigram)씩 묶어서 특징으로 사용
    min_df=3,                   # 단어가 전체 문서 중 최소 3번 이상 나타나야 특징으로 인정
    max_df=0.9                    # 단어가 전체 문서 중 90% 이상 나타나면 너무 흔하므로 제외
)

# --- 3. 모델 학습 및 변환 ---
# 훈련 데이터의 'document'로 TF-IDF 모델을 학습(fit).
# 이 과정에서 모델은 단어 사전을 만들고, 각 단어의 IDF 값을 계산함.
tfidf.fit(nsmc_train_df["document"])
# 학습된 모델을 사용해 훈련 데이터를 TF-IDF 벡터로 변환(transform).
nsmc_train_tfidf = tfidf.transform(nsmc_train_df["document"])

# --- 4. 학습된 TF-IDF 모델 저장 ---
# 나중에 재사용하기 위해 학습된 tfidf 객체를 pickle을 이용해 파일로 저장
with open("./tfidf_model.pkl", "wb") as file:
    pickle.dump(tfidf, file)
```

#### **2. 해당 설명**

**특징 벡터화(Feature Vectorization)**는 텍스트 마이닝의 심장과도 같습니다. 이 파트의 핵심은 **TF-IDF**입니다.

*   **TF (Term Frequency, 단어 빈도):** 한 문서 안에서 특정 단어가 얼마나 많이 나오는가? (예: '최고'라는 단어가 많이 나오면 그 리뷰는 '최고'와 관련이 깊음)
*   **IDF (Inverse Document Frequency, 역문서 빈도):** 특정 단어가 여러 문서에 걸쳐 얼마나 흔하게 나오는가? (예: '영화'라는 단어는 모든 영화 리뷰에 나오므로 변별력이 낮음 -> 가중치 감소. '인생'이라는 단어는 가끔 나오므로 변별력이 높음 -> 가중치 증가)

**TF-IDF = TF * IDF**. 즉, "특정 문서에는 자주 나오지만, 다른 모든 문서에서는 잘 안 나오는 단어"일수록 높은 점수를 받아 핵심 단어로 부각됩니다.

`Konlpy Okt`를 `tokenizer`로 사용하는 이유는, 한국어는 '조사'나 '어미'가 발달하여 "영화가"와 "영화는"을 다른 단어로 인식하는 문제를 해결하기 위함입니다. 형태소 분석기는 이들을 '영화'라는 동일한 어근으로 인식시켜 분석의 정확도를 높입니다. `ngram_range=(1, 2)`는 "재미"뿐만 아니라 "재미 없다"까지 하나의 특징으로 잡아내어 부정의 의미를 더 명확하게 포착하게 해줍니다.

마지막으로, 이렇게 학습된 `tfidf` 모델을 `pickle`로 저장하는 것은 매우 중요합니다. 이를 통해 나중에 새로운 데이터를 예측할 때, 훈련 데이터와 **완벽하게 동일한 기준**으로 벡터 변환을 수행할 수 있습니다.

#### **3. 응용 가능한 예제**

**"검색 엔진의 문서 순위 결정"**

사용자가 '파이썬 데이터 분석'이라고 검색했을 때, TF-IDF를 이용하면 '파이썬', '데이터', '분석'이라는 단어의 TF-IDF 점수가 높은 문서들을 찾아내어 검색 결과 상위에 노출시킬 수 있습니다.

#### **4. 추가하고 싶은 내용 (CountVectorizer vs TfidfVectorizer)**

`CountVectorizer`는 단순히 단어의 출현 횟수(TF)만을 세는 반면, `TfidfVectorizer`는 단어의 중요도(IDF)까지 고려합니다. 일반적으로 문서 분류나 감성 분석에서는 단순히 많이 나온 단어보다 문서의 주제를 잘 나타내는 핵심 단어를 찾는 것이 중요하므로 `TfidfVectorizer`가 더 좋은 성능을 보이는 경우가 많습니다.

#### **5. 심화 내용 (Word Embedding: Word2Vec, GloVe)**

TF-IDF는 단어의 순서나 문맥적 의미를 파악하지 못하는 한계가 있습니다. (예: '정말 좋은 영화'와 '좋은 정말 영화'를 동일하게 취급). **Word2Vec**이나 **GloVe**와 같은 **워드 임베딩(Word Embedding)** 기법은 단어의 의미 자체를 벡터 공간에 표현하여 '왕' - '남자' + '여자' = '여왕'과 같은 의미론적 관계까지 학습할 수 있는 더 진보된 벡터화 방식입니다.

---

### **Part 3: 감성 분석 모델 학습 및 최적화**

숫자 벡터로 변환된 데이터를 이용해 긍정과 부정을 구분하는 분류 모델을 만들고, 성능을 극대화하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# --- 1. 모델 생성 ---
# 로지스틱 회귀 모델 객체 생성. 이진 분류에 널리 사용됨
SA_lr = LogisticRegression(random_state=0)

# --- 2. 모델 학습 ---
# TF-IDF로 변환된 훈련 데이터(nsmc_train_tfidf)와 정답 라벨(nsmc_train_df['label'])로 모델을 학습
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])

# --- 3. 모델 최적화 (GridSearchCV) ---
# 테스트할 하이퍼파라미터 'C'의 후보 값들을 정의
# 'C'는 로지스틱 회귀의 규제(Regularization) 강도를 조절. 값이 작을수록 규제가 강해짐
params = {"C": [1, 3, 3.5, 4, 4.5, 5]}

# GridSearchCV 객체 생성
# SA_lr: 최적화할 모델
# param_grid=params: 테스트할 파라미터 조합
# cv=3: 3-겹 교차 검증(3-fold cross-validation) 수행
# scoring="accuracy": 성능 평가 지표로 '정확도'를 사용
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring="accuracy", verbose=1)

# GridSearchCV를 이용한 모델 학습. 이 과정에서 모든 C값과 교차 검증을 조합하여 최적의 C를 찾음
SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df["label"])

# --- 4. 최적 결과 확인 및 저장 ---
# 찾은 최적의 C값과 그때의 최고 점수를 출력
print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))

# 가장 성능이 좋았던 모델을 best_estimator_ 속성으로 가져옴
SA_lr_best = SA_lr_grid_cv.best_estimator_

# 최적화된 최종 모델을 pickle로 저장
with open("./SA_lr_best.pkl", "wb") as file:
    pickle.dump(SA_lr_best, file)
```

#### **2. 해당 설명**

이 파트에서는 분류 모델을 만들고 **하이퍼파라미터 튜닝(Hyperparameter Tuning)**을 통해 성능을 개선합니다.

*   **로지스틱 회귀:** '예' 또는 '아니오'와 같은 두 가지 경우를 분류하는 데 효과적인 알고리즘입니다. 각 단어의 TF-IDF 점수가 긍정/부정에 미치는 영향(가중치)을 학습합니다.
*   **GridSearchCV:** 모델의 성능을 결정하는 중요한 값인 **하이퍼파라미터**의 최적 조합을 자동으로 찾아주는 강력한 도구입니다. 개발자가 `params`에 후보 값들을 지정해주면, `GridSearchCV`는 모든 조합을 **교차 검증(Cross-Validation)** 방식으로 테스트하여 가장 성능이 좋은 조합을 알려줍니다.
*   **교차 검증(Cross-Validation):** 훈련 데이터를 여러 개(여기서는 `cv=3`이므로 3개)의 그룹으로 나눈 뒤, 하나를 검증용으로 사용하고 나머지를 훈련용으로 사용하는 과정을 반복하여 모델의 일반화 성능을 안정적으로 평가하는 방법입니다. 이를 통해 모델이 훈련 데이터에만 과적합(overfitting)되는 것을 방지합니다.

이 과정을 통해 우리는 "C값이 3.5일 때 이 모델의 성능이 가장 좋더라"라는 데이터 기반의 결론을 얻고, 이 최적의 모델을 최종 모델로 선택하여 저장합니다.

#### **3. 응용 가능한 예제**

**"고객 이탈 예측 모델 튜닝"**

랜덤 포레스트 모델을 사용하여 고객 이탈 여부를 예측할 때, `GridSearchCV`를 이용해 '트리의 개수(`n_estimators`)'와 '트리의 최대 깊이(`max_depth`)' 등 주요 하이퍼파라미터의 최적 조합을 찾아 모델의 예측 정확도를 극대화할 수 있습니다.

#### **4. 추가하고 싶은 내용 (다른 분류 모델)**

로지스틱 회귀 외에도 나이브 베이즈(Naive Bayes), 서포트 벡터 머신(SVM), 랜덤 포레스트(Random Forest), 그리고 최근에는 LSTM이나 BERT와 같은 딥러닝 모델들이 감성 분석에 널리 사용됩니다. 각 모델은 장단점이 있으므로 데이터의 특성에 맞게 선택하는 것이 중요합니다.

#### **5. 심화 내용 (파이프라인 구축)**

`scikit-learn`의 `Pipeline` 기능을 사용하면 **벡터화(TfidfVectorizer)와 모델 학습(LogisticRegression) 과정을 하나의 객체로** 묶을 수 있습니다. 이는 코드를 더 간결하게 만들고, 교차 검증 시 발생할 수 있는 데이터 유출(Data Leakage) 문제를 원천적으로 방지하여 더 신뢰성 있는 모델을 구축하는 데 도움이 됩니다.

---

### **Part 4: 모델 평가 및 새로운 데이터에 적용**

잘 만들어진 모델의 성능을 최종 검증하고, 이를 실제 문제(코로나 뉴스 분석)에 적용하여 인사이트를 도출하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from sklearn.metrics import accuracy_score

# --- 1. 모델 평가 ---
# 저장해둔 TF-IDF 모델로 평가용 데이터(test set)를 벡터화
# 주의: fit_transform이 아닌 transform을 사용! (훈련 데이터의 기준으로 변환)
nsmc_test_tfidf = tfidf.transform(nsmc_test_df["document"])

# 최적화된 모델로 예측 수행
test_predict = SA_lr_best.predict(nsmc_test_tfidf)

# 정확도 계산: 실제 라벨(nsmc_test_df["label"])과 예측값(test_predict)을 비교
print("감성 분석 정확도 : ", round(accuracy_score(nsmc_test_df["label"], test_predict), 3))

# --- 2. 실시간 예측 ---
st = input("감성 분석하기위한 문장을 입력 해주세요: ")
# ... (입력 문장 전처리) ...
st_tfidf = tfidf.transform(st) # 동일한 tfidf 모델로 변환
st_predict = SA_lr_best.predict(st_tfidf) # 동일한 감성분석 모델로 예측
# ... (결과 출력) ...

# --- 3. 코로나 뉴스에 적용 ---
# ... (뉴스 데이터 로드 및 전처리) ...
# 뉴스의 'title'과 'description'을 각각 벡터화하고 감성 예측
data_title_tfidf = tfidf.transform(data_df["title"])
data_title_predict = SA_lr_best.predict(data_title_tfidf)
# ... (description도 동일하게 수행) ...
data_df["title_label"] = data_title_predict
data_df["description_label"] = data_description_predict

# --- 4. 결과 분석 및 시각화 ---
# 긍정/부정 뉴스에서 각각 명사만 추출하여 가장 많이 나온 단어 분석
# ... (okt.nouns()를 이용한 명사 추출) ...
# ... (TfidfVectorizer로 단어별 중요도 다시 계산) ...
# ... (matplotlib을 이용해 상위 단어 바 차트 시각화) ...
```

#### **2. 해당 설명**

이 파트에서는 학습된 모델의 **실용성**을 보여줍니다.

1.  **성능 검증:** 한 번도 본 적 없는 평가용 데이터(test set)에 대해 약 85%의 정확도를 보임으로써, 이 모델이 특정 데이터에만 국한되지 않고 **일반적인 성능**을 가짐을 입증합니다.
2.  **새로운 데이터 적용:** 완전히 다른 도메인인 **코로나 뉴스**에 영화 리뷰로 학습된 모델을 적용합니다. 이는 모델이 특정 분야를 넘어 **범용적인 긍정/부정 어조를 학습**했음을 전제로 합니다. (물론, 도메인이 다르면 성능이 저하될 수 있습니다.)
3.  **인사이트 도출:** 단순히 긍정/부정을 나누는 데서 그치지 않고, "부정적인 뉴스에서는 어떤 단어들이 주로 언급되는가?", "긍정적인 뉴스에서는 어떤 단어들이 언급되는가?"를 분석합니다. 바 차트 시각화를 통해 '확진자', '사망' 같은 단어가 부정 뉴스에서, '백신', '치료제' 같은 단어가 긍정 뉴스에서 두드러지는 경향을 한눈에 파악할 수 있습니다. 이는 텍스트 데이터를 통해 **사회적 여론이나 관심사를 파악**하는 텍스트 마이닝의 강력한 활용 예시입니다.

#### **3. 응용 가능한 예제**

**"콜센터 상담 기록 분석"**

고객과의 전화 상담을 텍스트로 변환한 데이터에 감성 분석 모델을 적용하여 '불만 상담'과 '만족 상담'을 자동으로 분류할 수 있습니다. 이후, 불만 상담에서 자주 등장하는 키워드('오류', '지연', '불친절' 등)를 분석하여 서비스 개선점을 도출할 수 있습니다.

#### **4. 추가하고 싶은 내용 (도메인 특화 모델)**

영화 리뷰로 학습한 모델을 뉴스에 적용하는 것은 범용성을 테스트하는 좋은 예시이지만, 최고의 성능을 위해서는 해당 도메인(뉴스, 상품평, 의료 기록 등)의 데이터로 모델을 다시 학습시키는 **도메인 특화(Domain-specific) 튜닝**이 필요합니다.

#### **5. 심화 내용 (감성 사전 활용)**

모델 기반 감성 분석 외에, 단어마다 긍정/부정 점수가 미리 정의된 **감성 사전(Sentiment Lexicon)**을 활용하는 방법도 있습니다. 문장에 포함된 단어들의 점수를 합산하거나 평균 내어 감성을 판단하는 방식으로, 학습 데이터가 부족할 때 유용하게 사용될 수 있습니다.

---

### **Part 5: 토픽 모델링 (LDA) 및 시각화**

코로나 뉴스 데이터에 숨어있는 주요 주제들을 자동으로 발견하고 시각화하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import gensim
import gensim.corpora as corpora
import pyLDAvis.gensim

# --- 1. LDA용 데이터 준비 ---
# 뉴스 본문('description')에서 명사만 추출 (토픽은 주로 명사로 표현되므로)
description_noun_tk = [okt.nouns(d) for d in description]
# 한 글자짜리 명사는 의미가 없는 경우가 많아 제거
description_noun_tk2 = [[i for i in d if len(i) > 1] for d in description_noun_tk]

# --- 2. 사전 및 코퍼스 생성 ---
# 각 명사에 고유한 ID를 부여한 사전을 생성
dictionary = corpora.Dictionary(description_noun_tk2)
# 각 문서를 (단어ID, 출현빈도) 형태로 변환한 코퍼스(말뭉치)를 생성
corpus = [dictionary.doc2bow(word) for word in description_noun_tk2]

# --- 3. LDA 모델 학습 ---
# k: 추출할 토픽의 개수 (분석가가 지정)
k = 4
# gensim의 LdaMulticore 모델을 이용해 학습
lda_model = gensim.models.ldamulticore.LdaMulticore(
    corpus=corpus, iterations=12, num_topics=k, id2word=dictionary, passes=1, workers=10
)

# 학습된 각 토픽과, 그 토픽을 구성하는 주요 단어들을 출력
print(lda_model.print_topics(num_topics=k, num_words=15))

# --- 4. 시각화 ---
# pyLDAvis를 이용해 LDA 결과를 인터랙티브하게 시각화할 준비
lda_vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
# 시각화 결과 출력 (Jupyter Notebook 환경에서 인터랙티브하게 보임)
pyLDAvis.display(lda_vis)
```

#### **2. 해당 설명**

**토픽 모델링**은 감성 분석과는 다른 차원의 분석입니다. "이 글이 긍정인가 부정인가?"가 아니라 **"이 글이 대체 무엇에 대한 이야기인가?"**를 알아내는 비지도 학습 기법입니다.

*   **LDA (Latent Dirichlet Allocation):** 문서들은 여러 개의 잠재적인 토픽(주제)들의 혼합으로 이루어져 있다는 가정하에, 문서 집합으로부터 토픽들을 자동으로 추출하는 알고리즘입니다.
*   **과정:** LDA는 (1) 각 문서가 어떤 토픽 비율을 가질지, (2) 각 토픽이 어떤 단어 비율을 가질지를 확률적으로 추론합니다. 그 결과, `print_topics`에서 보듯이 "토픽 0은 '백신', '접종', '코로나' 등의 단어로 이루어져 있다"와 같은 결과를 보여줍니다.
*   **결과 해석:** 분석가는 각 토픽을 구성하는 단어들을 보고 "아, 토픽 0은 **백신 및 예방 접종**에 대한 주제구나"라고 의미를 부여할 수 있습니다.
*   **pyLDAvis 시각화:**
    *   **좌측 (토픽 간 관계):** 각 원은 하나의 토픽을 나타냅니다. 원의 크기는 전체 문서에서 해당 토픽이 차지하는 비중을, 원 사이의 거리는 토픽 간의 유사도를 의미합니다. (가까울수록 유사한 토픽)
    *   **우측 (토픽 내 단어):** 좌측에서 특정 토픽(원)을 클릭하면, 해당 토픽을 구성하는 주요 단어들이 중요도 순으로 막대그래프에 나타납니다.

이를 통해 수백, 수천 건의 뉴스 기사를 일일이 읽지 않고도, 전체 뉴스 데이터에서 다루어지는 핵심 주제들과 그 비중을 빠르고 직관적으로 파악할 수 있습니다.

#### **3. 응용 가능한 예제**

**"학술 논문 데이터베이스 분석"**

수만 건의 학술 논문 초록(abstract) 데이터를 LDA로 분석하여, 최근 몇 년간 해당 분야에서 어떤 연구 주제들이 떠오르고 있으며, 어떤 주제들이 서로 연관되어 연구되고 있는지 트렌드를 파악할 수 있습니다.

#### **4. 추가하고 싶은 내용 (최적의 토픽 수 K 찾기)**

K-평균 군집 분석과 마찬가지로, LDA에서도 최적의 토픽 수 `k`를 찾는 것이 중요합니다. `gensim`의 `CoherenceModel`을 사용하면, 다양한 `k`값에 대해 **토픽 응집도 점수(Coherence Score)**를 계산하여 가장 의미론적으로 일관된 토픽을 생성하는 최적의 `k`를 찾는 데 도움을 받을 수 있습니다.

#### **5. 심화 내용 (LDA의 활용)**

LDA는 단순히 주제를 찾는 것을 넘어, 각 문서가 어떤 토픽에 몇 퍼센트씩 속하는지 알려주기 때문에 이를 **문서의 특징 벡터**로 활용할 수도 있습니다. 예를 들어, LDA로 추출한 토픽 분포를 입력으로 사용하여 문서를 특정 카테고리로 분류하는 모델의 성능을 향상시키는 데 사용할 수 있습니다.