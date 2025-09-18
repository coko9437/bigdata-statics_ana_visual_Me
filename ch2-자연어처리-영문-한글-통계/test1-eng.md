
---

### **Ch2-1 코드 종합 설명**

이 코드는 여러 엑셀 파일에 나뉘어 저장된 학술 문서 제목 데이터를 하나로 합친 후, **자연어 처리(NLP)** 기술을 이용해 텍스트를 정제하고 핵심 단어를 추출하는 과정을 담고 있습니다. 최종적으로는 추출된 단어의 빈도를 계산하여 **막대 그래프**와 **워드 클라우드**로 시각화함으로써 'Big data' 관련 연구의 핵심 키워드와 연도별 연구 동향을 직관적으로 파악하는 것을 목표로 합니다.

전체적인 흐름은 다음과 같습니다.

1.  **데이터 수집 및 병합**: 여러 개의 엑셀 파일을 자동으로 찾아 하나의 데이터 프레임으로 통합합니다.
2.  **텍스트 전처리**: 분석에 불필요한 기호를 제거하고, 단어를 소문자로 통일하며, 의미 없는 단어(불용어)를 제거하고, 단어를 기본형으로 바꾸는 등 텍스트를 분석 가능한 형태로 가공합니다.
3.  **데이터 탐색 및 분석**: 전처리된 단어들의 빈도를 계산하여 가장 많이 등장하는 핵심 키워드를 파악하고, 연도별 논문 수를 집계하여 연구 트렌드를 분석합니다.
4.  **시각화**: 분석 결과를 막대 그래프, 꺾은선 그래프, 그리고 이미지 모양을 활용한 워드 클라우드로 표현하여 인사이트를 효과적으로 전달합니다.

---

### **Part 1: 데이터 수집 및 병합**

분석의 첫 단계로, 여러 곳에 흩어져 있는 데이터를 프로그래밍을 통해 자동으로 불러오고 하나로 합치는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# 관련 패키지들 임포트 하기.
import pandas as pd
# 파일 경로를 다룰 때 사용
import glob
# ... (다른 import 문 생략) ...

# 현재 폴더에 있는 'exportExcelData_*.xls' 패턴의 모든 파일 경로를 리스트로 가져옴
# '*'는 와일드카드로, 어떤 문자열이든 매칭됨 (예: exportExcelData_1.xls, exportExcelData_BigData.xls 등)
all_files = glob.glob("./exportExcelData_*.xls")

# 여러 데이터프레임을 담을 빈 리스트 생성
all_files_data = []

# all_files 리스트에 있는 각 파일 경로에 대해 반복 작업 수행
for file in all_files:
    # pd.read_excel() 함수로 엑셀 파일을 읽어 데이터프레임(표 형태)으로 변환
    data_frame = pd.read_excel(file)
    # 읽어온 데이터프레임을 all_files_data 리스트에 추가
    all_files_data.append(data_frame)

# 여러 데이터프레임이 담긴 리스트를 하나의 데이터프레임으로 합침
# axis=0: 위아래(수직) 방향으로 데이터를 이어 붙임
# ignore_index=True: 기존 파일들의 인덱스를 무시하고, 0부터 시작하는 새로운 인덱스를 생성
all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)

# 합쳐진 데이터프레임의 행과 열 개수를 확인
print(f"all_files_data_concat.shape : {all_files_data_concat.shape}")

# 최종 병합된 데이터를 새로운 CSV 파일로 저장
# encoding="utf-8": 한글 깨짐을 방지하기 위한 인코딩 설정
# index=False: 데이터프레임의 인덱스는 파일에 저장하지 않음
all_files_data_concat.to_csv("./riss_Bigdata_2025.csv", encoding="utf-8", index=False)
```

#### **2. 해당 설명**

데이터 분석의 시작은 데이터를 불러오는 것입니다. 위 코드는 `glob` 라이브러리를 사용해 특정 패턴(`exportExcelData_*.xls`)을 가진 모든 파일의 목록을 한 번에 가져옵니다. 수작업으로 파일 10개를 일일이 불러오는 대신, 이처럼 자동화하면 파일이 수십, 수백 개로 늘어나도 코드를 수정할 필요가 없어 매우 효율적입니다. `for` 반복문을 통해 각 엑셀 파일을 `pandas` 데이터프레임으로 변환하여 리스트에 차곡차곡 쌓고, 마지막에 `pd.concat` 함수로 모든 데이터프레임을 합쳐 거대한 단일 데이터셋을 완성합니다. 이 과정은 여러 소스에서 수집된 데이터를 통합하여 전체적인 그림을 보는 데 필수적입니다.

#### **3. 응용 가능한 예제**

**"전국 매장의 일일 매출 엑셀 파일을 취합하여 월간 전체 실적 분석하기"**

각 매장에서 `서울_0901_sales.xlsx`, `부산_0901_sales.xlsx` 와 같이 매일 생성되는 매출 파일을 `glob.glob("./*_sales.xlsx")` 코드로 모두 불러온 뒤, `pd.concat`으로 합쳐 전사적인 매출 동향을 분석하는 데 사용할 수 있습니다.

#### **4. 추가하고 싶은 내용 (오류 처리)**

만약 엑셀 파일 중 일부가 손상되었거나 형식이 다를 경우, `pd.read_excel(file)` 부분에서 오류가 발생하여 프로그램 전체가 멈출 수 있습니다. 이를 방지하기 위해 `try-except` 구문을 사용하여 오류가 발생하더라도 해당 파일을 건너뛰고 다음 파일 처리를 계속하도록 코드를 더 안정적으로 만들 수 있습니다.

```python
for file in all_files:
    try:
        data_frame = pd.read_excel(file)
        all_files_data.append(data_frame)
    except Exception as e:
        print(f"{file}을 읽는 중 오류 발생: {e}") # 오류가 난 파일과 원인을 출력
```

#### **5. 심화 내용 (데이터 파이프라인 개념)**

이처럼 데이터를 **추출(Extract)**하고, (뒤에서 다룰) **변환(Transform)**하며, 최종적으로 파일이나 데이터베이스에 **적재(Load)**하는 일련의 과정을 **ETL(Extract, Transform, Load) 파이프라인**이라고 합니다. 현재 코드는 이 ETL 파이프라인의 가장 기본적인 형태이며, 실제 현업에서는 Airflow나 Prefect 같은 전문 도구를 사용해 이러한 데이터 처리 과정을 자동화하고 관리합니다.

---

### **Part 2: 텍스트 전처리 (자연어 처리)**

수집된 원본 텍스트(논문 제목)를 분석에 적합한 형태로 깨끗하게 다듬는, 자연어 처리의 핵심 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# ... (import 및 데이터 로드 생략) ...
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# NLTK의 영어 불용어(the, a, is 등) 사전을 불러와 set 형태로 저장
# set은 중복을 허용하지 않으며, 특정 요소가 포함되어 있는지 확인할 때 리스트보다 훨씬 빠름
stopWords = set(stopwords.words("english"))

# 표제어 추출을 위한 WordNetLemmatizer 객체 생성
lemma = WordNetLemmatizer()

# 정제된 단어들을 담을 리스트 초기화
words = []

# '제목' 열의 모든 제목에 대해 반복 처리
for title in all_title:
    # 1. 정제 (Cleaning): 정규 표현식 사용
    # [^a-zA-Z]+ : a-z, A-Z (모든 영어 알파벳)가 아닌(+) 모든 문자(^)를
    # " " (공백 한 칸)으로 치환(sub)
    enWords = re.sub(r"[^a-zA-Z]+", " ", str(title))

    # 2. 정규화 (Normalization) & 토큰화 (Tokenization)
    # .lower(): 모든 알파벳을 소문자로 변환
    # word_tokenize(): 문자열을 공백, 구두점 등을 기준으로 잘라 단어 리스트(토큰)로 만듦
    enWordsToken = word_tokenize(enWords.lower())

    # 3. 불용어 제거 (Stopword Removal)
    # enWordsToken 리스트의 각 단어(w)가 stopWords 집합에 포함되어 있지 않은 경우에만
    # 새로운 리스트에 포함시킴 (리스트 컴프리헨션 문법)
    enWordsTokenStop = [w for w in enWordsToken if w not in stopWords]

    # 4. 표제어 추출 (Lemmatization)
    # enWordsTokenStop 리스트의 각 단어(w)에 대해 lemma.lemmatize() 함수를 적용
    # 'studies', 'studying' -> 'study' 와 같이 단어의 원형을 찾아줌
    enWordsTokenStopLemma = [lemma.lemmatize(w) for w in enWordsTokenStop]

    # 최종 처리된 단어 리스트를 words 리스트에 추가 (결과는 리스트의 리스트 형태가 됨)
    words.append(enWordsTokenStopLemma)

# 2차원 리스트 -> 1차원 리스트로 변환
from functools import reduce
result = reduce(lambda x, y: x + y, words)
# 예: [[a,b], [c,d]] -> [a,b,c,d]
```

#### **2. 해당 설명**

텍스트 데이터는 컴퓨터가 바로 이해할 수 없는 비정형 데이터이므로, 분석 전에 반드시 '전처리' 과정이 필요합니다. 위 코드는 자연어 처리의 표준적인 전처리 5단계를 체계적으로 수행합니다.
`re.sub`으로 영어 알파벳 외의 모든 기호를 제거하고(`정제`), `lower()`로 대소문자를 통일하며(`정규화`), `word_tokenize`로 문장을 단어 단위로 쪼갭니다(`토큰화`). 그 후 `stopwords`를 이용해 'a', 'the'처럼 의미는 없지만 자주 등장하는 단어들을 걸러내고(`불용어 제거`), 마지막으로 `WordNetLemmatizer`를 통해 'analyzing', 'analyzed' 같은 다양한 형태의 단어를 'analyze'라는 기본형(`표제어`)으로 통일합니다. 이 과정을 거쳐야만 정확한 단어 빈도 분석이 가능해집니다.

#### **3. 응용 가능한 예제**

**"온라인 쇼핑몰 상품평 분석을 통한 고객 감성 분석"**

고객들이 남긴 상품평 텍스트에서 긍정, 부정 키워드를 추출하기 전에, 위와 동일한 전처리 과정을 거쳐야 합니다. 예를 들어 "이 제품 정말 최고!!! 강추합니다bb" 라는 텍스트에서 특수문자(`!`, `b`)를 제거하고, '합니다' 같은 불용어를 처리해야 '제품', '최고', '강추'와 같은 핵심적인 의미를 가진 단어만 남길 수 있습니다.

#### **4. 추가하고 싶은 내용 (커스텀 불용어 추가)**

분석 목적에 따라 기본 불용어 사전에 없는 단어를 추가해야 할 때가 있습니다. 예를 들어, 'Big data' 논문 분석에서는 'study', 'research', 'paper', 'using' 같은 단어가 너무 자주 등장하여 분석에 방해가 될 수 있습니다. 이럴 때는 다음과 같이 직접 불용어를 추가할 수 있습니다.

```python
stopWords = set(stopwords.words("english"))
custom_stopwords = ['study', 'research', 'paper', 'using']
for word in custom_stopwords:
    stopWords.add(word)
```

#### **5. 심화 내용 (어간 추출 vs 표제어 추출)**

코드에서는 **표제어 추출(Lemmatization)**을 사용했습니다. 이와 비슷한 개념으로 **어간 추출(Stemming)**이 있습니다.
*   **어간 추출(Stemming)**: `studies`, `studying` -> `studi` 처럼 단순히 단어의 뒷부분을 잘라내어 원형을 찾습니다. 속도는 빠르지만, `studi`처럼 사전에 없는 단어가 될 수 있습니다. (예: PorterStemmer)
*   **표제어 추출(Lemmatization)**: `studies`, `studying` -> `study` 처럼 단어의 문법적 형태와 의미를 고려하여 사전에 있는 기본형 단어를 찾아줍니다. 속도는 느리지만 훨씬 정교합니다. (예: WordNetLemmatizer)

일반적으로 분석의 정확도가 중요하다면 표제어 추출을 사용하는 것이 더 좋은 결과를 낳습니다.

---

### **Part 3: 데이터 탐색 및 시각화 (막대 그래프, 꺾은선 그래프)**

전처리를 마친 데이터를 바탕으로 실제 의미 있는 정보를 추출하고, 이를 그래프로 표현하여 경향성을 파악하는 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from collections import Counter
import matplotlib.pyplot as plt

# 1차원 단어 리스트(result)를 Counter 객체에 넣어 단어별 빈도를 계산
# 결과는 {'data': 1235, 'big': 1113, ...} 와 같은 딕셔너리 형태
count = Counter(result)

# 빈도가 가장 높은 상위 50개 단어를 추출
# most_common(50)은 (단어, 빈도) 형태의 튜플 리스트를 반환
word_count = dict()
for tag, counts in count.most_common(50):
    # 단어 길이가 1 이하인 경우(a, b 등)는 제외
    if (len(str(tag)) > 1):
        word_count[tag] = counts

# --- 막대 그래프 시각화 ---
# word_count 딕셔너리를 값(빈도수) 기준으로 내림차순 정렬
# key=word_count.get: 딕셔너리의 값을 정렬 기준으로 사용
sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
# xticks: x축의 눈금 위치와 라벨을 설정
# rotation="vertical": x축 라벨(단어)을 세로로 표시하여 겹치지 않게 함
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation="vertical")
plt.show()


# --- 꺾은선 그래프 시각화 (연도별 논문 수) ---
# '출판일'을 기준으로 데이터를 그룹화하고, 각 그룹의 데이터 개수를 셈
summary_date = all_files_data_concat.groupby("출판일", as_index=False)["doc_count"].count()

plt.figure(figsize=(12, 5)) # 그래프 크기 설정
plt.xlabel("year")          # x축 이름
plt.ylabel("doc_count")    # y축 이름
plt.grid(True)              # 격자 표시

# plot: 꺾은선 그래프를 그림
plt.plot(range(len(summary_date)), summary_date["doc_count"])
# x축의 숫자 눈금(0, 1, 2...)을 실제 연도 텍스트로 변경
plt.xticks(range(len(summary_date)), [text for text in summary_date["출판일"]])
plt.show()
```

#### **2. 해당 설명**

데이터를 숫자로 요약하는 것도 중요하지만, 시각화는 숨겨진 패턴을 발견하는 가장 효과적인 방법입니다. `collections` 라이브러리의 `Counter`는 리스트 안의 각 요소가 몇 번씩 나타나는지 순식간에 계산해주는 매우 편리한 도구입니다. `most_common()`을 통해 핵심 키워드 50개를 추출하고, `matplotlib`을 이용해 **막대 그래프**로 시각화하면 어떤 단어들이 가장 중요한지 명확하게 알 수 있습니다.
또한, `pandas`의 `groupby` 기능은 특정 열(여기서는 '출판일')을 기준으로 데이터를 요약하는 데 매우 강력합니다. 연도별 논문 수를 계산하고 이를 **꺾은선 그래프**로 시각화함으로써, 'Big data' 분야에 대한 연구 관심도가 시간이 지남에 따라 어떻게 변했는지(증가, 감소, 정체 등) 그 추세를 한눈에 파악할 수 있습니다.

#### **3. 응용 가능한 예제**

**"웹사이트 로그 분석을 통한 시간대별 사용자 접속 현황 시각화"**

사용자 접속 기록 데이터에서 '접속 시간' 열을 `groupby`하여 시간대별 접속자 수를 계산한 뒤, 막대 그래프나 꺾은선 그래프로 시각화하여 어느 시간대에 사용자가 가장 활발한지(피크 타임) 분석할 수 있습니다.

#### **4. 추가하고 싶은 내용 (가독성 높은 수평 막대 그래프)**

x축 라벨이 길어서 세로로 표시해도 가독성이 떨어질 경우, 수평 막대 그래프(`plt.barh`)를 사용하는 것이 더 효과적일 수 있습니다. y축에 단어를 표시하므로 글자가 겹칠 염려가 없습니다.

```python
# sorted_Keys와 sorted_Values를 오름차순으로 변경해야 위쪽이 가장 큰 값이 됨
sorted_Keys.reverse()
sorted_Values.reverse()
plt.barh(sorted_Keys, sorted_Values)
plt.show()
```

#### **5. 심화 내용 (N-gram 분석)**

현재 분석은 'data', 'big'처럼 한 단어(uni-gram)씩 분석했습니다. 하지만 'big data', 'data analytics'처럼 두 단어 이상이 합쳐져야 의미가 명확해지는 경우가 많습니다. 이렇게 연속된 n개의 단어를 묶어서 분석하는 것을 **N-gram 분석**이라고 합니다. `nltk` 라이브러리는 `bigrams` (2-gram), `trigrams` (3-gram) 등을 쉽게 생성하는 기능을 제공하여 더 깊이 있는 키워드 분석을 가능하게 합니다.

---

### **Part 4: 워드 클라우드 시각화**

분석의 하이라이트로, 단어의 빈도를 글자의 크기로 표현하여 가장 핵심적인 주제를 시각적으로 강조하는 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np

# 워드 클라우드를 그릴 모양의 마스크 이미지 파일을 불러옴
# Image.open(): PIL 라이브러리로 이미지 파일을 열어 이미지 객체로 만듦
# np.array(): 이미지 객체를 숫자로 이루어진 numpy 배열로 변환.
#             WordCloud는 이 배열의 모양대로 단어를 배치함.
alice_mask = np.array(Image.open("convert_test1.png"))

# WordCloud 객체 생성 및 주요 옵션 설정
# background_color="white": 배경색을 흰색으로 지정
# max_words=1000: 워드 클라우드에 표시할 최대 단어 개수
# mask=alice_mask: 단어를 채워 넣을 모양(틀)으로 위에서 만든 이미지 배열을 사용
wc = WordCloud(background_color="white", max_words=1000, mask=alice_mask)

# generate_from_frequencies(): 단어 빈도 정보가 담긴 딕셔너리(word_count)를 받아
#                                빈도수에 비례하여 글자 크기가 다른 워드 클라우드를 생성
cloud = wc.generate_from_frequencies(word_count)

# matplotlib을 이용해 최종 생성된 워드 클라우드 이미지를 화면에 표시
plt.figure(figsize=(8, 8)) # 출력될 이미지의 크기 설정
plt.imshow(cloud)         # 이미지 표시
plt.axis("off")           # x, y 축 및 눈금선을 보이지 않게 처리
plt.show()
```

#### **2. 해당 설명**

**워드 클라우드**는 텍스트 데이터 분석 결과를 비전문가도 쉽게 이해할 수 있도록 만드는 매우 효과적인 시각화 기법입니다. `wordcloud` 라이브러리는 이 과정을 매우 간단하게 만들어 줍니다. `WordCloud` 객체를 생성할 때 배경색, 최대 단어 수 등 다양한 옵션을 지정할 수 있습니다. 특히 이 코드의 핵심은 `mask` 옵션입니다. `PIL`과 `numpy`를 이용해 특정 이미지 파일을 숫자 배열로 변환하고, 이를 `mask`로 지정하면 워드 클라우드가 단순한 사각형이 아닌 원하는 이미지 모양으로 생성됩니다. 마지막으로 `generate_from_frequencies` 메소드에 앞서 계산한 단어 빈도 딕셔너리를 전달하면, 빈도가 높은 단어는 크게, 낮은 단어는 작게 배치된 아름다운 워드 클라우드가 완성됩니다.

#### **3. 응용 가능한 예제**

**"회사 로고 모양으로 핵심 가치 워드 클라우드 만들기"**

회사의 비전이나 직원 설문조사 결과를 텍스트 분석한 후, 핵심 키워드를 추출하여 회사 로고 이미지 파일을 마스크로 사용해 워드 클라우드를 만들 수 있습니다. 이는 회사 소개 자료나 내부 커뮤니케이션 자료로 매우 인상 깊게 활용될 수 있습니다.

#### **4. 추가하고 싶은 내용 (색상 변경)**

워드 클라우드의 단어 색상이 단조롭다고 느껴진다면 `colormap` 옵션을 통해 다채로운 색상 조합을 적용할 수 있습니다. `matplotlib`에서 제공하는 다양한 컬러맵(예: 'viridis', 'plasma', 'inferno', 'magma', 'cividis')을 사용할 수 있습니다.

```python
# ... (alice_mask 생성 부분은 동일) ...
wc = WordCloud(background_color="white", max_words=1000, mask=alice_mask, colormap='viridis')
cloud = wc.generate_from_frequencies(word_count)
# ... (imshow 부분은 동일) ...
```

#### **5. 심화 내용 (워드 클라우드의 한계와 대안)**

워드 클라우드는 시각적으로 매우 뛰어나지만, 단어 간의 관계나 맥락을 보여주지 못한다는 명확한 한계가 있습니다. 예를 들어 'big'와 'data'가 항상 같이 등장하더라도, 워드 클라우드에서는 멀리 떨어져 표시될 수 있습니다. 이러한 한계를 극복하기 위해, 단어들의 관계를 네트워크(연결망) 형태로 보여주는 **네트워크 분석(Network Analysis)**이나, 문서에 숨겨진 주제들을 찾아내는 **토픽 모델링(Topic Modeling, 예: LDA)**과 같은 더 고차원적인 분석 기법을 함께 사용하면 텍스트 데이터로부터 훨씬 풍부한 인사이트를 얻을 수 있습니다.