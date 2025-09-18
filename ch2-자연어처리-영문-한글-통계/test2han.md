
---

### **Ch2-2 코드 종합 설명**

이 코드는 웹 크롤링으로 수집한 한국어 뉴스 데이터(JSON 형식)를 대상으로 **형태소 분석**을 수행하여, 텍스트에 담긴 핵심 명사를 추출하고 그 빈도를 분석하는 과정을 담고 있습니다. 영어 분석(ch2-1)과 달리, 조사가 발달한 한국어의 언어적 특성을 처리하기 위해 `KoNLPy`라는 전문적인 형태소 분석기 라이브러리를 사용하는 것이 가장 큰 특징입니다.

전체적인 흐름은 다음과 같습니다.

1.  **환경 설정의 이해**: 한글 자연어 처리에 필요한 Java(JDK), `JPype`, `KoNLPy` 설치 과정을 이해합니다. 이는 `KoNLPy`가 Java 기반으로 동작하기 때문입니다.
2.  **데이터 로딩 및 정제**: JSON 파일을 읽어와 분석에 필요한 텍스트 데이터('description')만 추출하고, 한글과 숫자를 제외한 불필요한 특수문자를 제거합니다.
3.  **형태소 분석**: `KoNLPy`의 `Okt` 형태소 분석기를 사용하여 전체 텍스트에서 **명사(Nouns)**만 정확하게 추출합니다.
4.  **빈도 분석 및 시각화**: 추출된 명사의 빈도를 계산하고, **막대 그래프**와 이미지 마스크를 적용한 **워드 클라우드**로 시각화합니다. 이 과정에서 한글 폰트가 깨지지 않도록 설정하는 중요한 작업이 포함됩니다.

---

### **Part 1: 복잡한 환경 설정의 이해**

코드를 실행하기에 앞서, 왜 이런 복잡한 준비 과정이 필요한지 이해하는 것이 중요합니다.

#### **1. 설명**

영어는 보통 띄어쓰기를 기준으로 단어를 분리(`토큰화`)해도 의미 파악이 어느 정도 가능합니다. 하지만 한국어는 **교착어**의 특성을 가져, '맛집**은**', '맛집**을**', '맛집**에서**'처럼 명사 뒤에 다양한 **조사(은/는/이/가 등)**가 붙습니다. 단순히 띄어쓰기로만 자르면 '맛집은'과 '맛집을'은 다른 단어로 인식되어 '맛집'이라는 핵심 키워드의 빈도를 정확히 셀 수 없습니다.

**형태소 분석기(`KoNLPy`)**는 바로 이 문제를 해결해 줍니다. 문장을 의미를 가진 가장 작은 단위인 **형태소**로 분해하고, 각 형태소의 품사(명사, 동사, 조사 등)를 알려줍니다. 예를 들어 '맛집은'을 '맛집(명사)'과 '은(조사)'으로 분리해주므로, 우리는 정확하게 '맛집'이라는 명사만 추출하여 분석할 수 있습니다.

`KoNLPy`의 여러 분석기(Okt, Kkma 등)는 내부적으로 Java 언어로 만들어져 있기 때문에, 파이썬에서 이를 사용하려면 다음과 같은 '다리' 역할의 도구들이 반드시 필요합니다.
*   **JDK (Java Development Kit)**: Java 프로그램의 실행 환경.
*   **JPype**: 파이썬이 Java 코드를 불러와 사용할 수 있게 해주는 연결 도구.
*   **환경 변수 설정**: 컴퓨터가 JDK와 파이썬의 위치를 언제 어디서든 찾을 수 있도록 경로를 알려주는 작업.

이러한 이유로 영어 분석보다 초기 설정 과정이 복잡합니다.

#### **2. 심화 내용 (가상 환경의 중요성)**

주석에 언급된 것처럼 특정 버전의 파이썬과 라이브러리가 필요한 경우, 프로젝트마다 다른 환경을 요구할 때가 많습니다. 이때 **가상 환경(`venv` 또는 `conda`)**을 사용하면 프로젝트별로 독립된 개발 환경을 구축할 수 있습니다. 예를 들어, A 프로젝트는 파이썬 3.10.7, B 프로젝트는 파이썬 3.12.0을 사용하도록 격리하여 버전 충돌 문제를 근본적으로 방지할 수 있습니다. 복잡한 설정을 할 때는 가상 환경을 사용하는 것이 매우 권장됩니다.

---

### **Part 2: 데이터 로딩 및 텍스트 정제**

분석할 원본 데이터를 불러와 필요한 부분만 추출하고, 기본적인 클리닝 작업을 수행하는 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import json
import re

# 분석할 JSON 파일 경로
inputFileName = "./부산서면맛집2_naver_news.json"
# 파일을 열고(open), 내용을 문자열로 읽은 뒤(.read()), json 형식으로 파싱(json.loads)
# "r"(읽기 모드), encoding="utf-8"(한글 깨짐 방지)
data = json.loads(open(inputFileName, "r", encoding="utf-8").read())

# 추출된 텍스트를 모두 합쳐서 담을 빈 문자열 변수
message = ""

# data는 딕셔너리들을 담고 있는 리스트. 각 딕셔너리(item)에 대해 반복.
for item in data:
    # 딕셔너리(item) 안에 'description'이라는 키(key)가 있는지 확인
    if "description" in item.keys():
        # 1. item["description"]으로 해당 키의 값(뉴스 본문)을 가져옴
        # 2. re.sub()로 정규 표현식에 해당하는 부분을 찾아 다른 문자로 치환
        #    r"[^\w]": \w(알파벳, 숫자, 밑줄_)가 아닌(^) 모든 문자를
        #    " "(공백 한 칸)으로 변경. -> 즉, 한글, 영어, 숫자만 남기고 특수문자는 공백으로.
        cleaned_text = re.sub(r"[^\w]", " ", item["description"])
        # 3. 정제된 텍스트를 message 변수에 계속 이어 붙임
        message = message + cleaned_text + " " # 각 기사 사이에 공백 추가
```

#### **2. 해당 설명**

`json.loads()` 함수는 JSON 형식의 텍스트 파일을 파이썬의 기본 자료구조(리스트, 딕셔너리)로 변환해주는 역할을 합니다. 이렇게 변환된 `data` 리스트를 `for` 문으로 순회하며, 각 기사(`item`)에서 내용이 담긴 `'description'` 부분만 추출합니다. `re.sub(r"[^\w]", " ", ...)`는 텍스트 정제의 핵심으로, `\w`에 포함되지 않는 각종 특수문자(마침표, 쉼표, 따옴표 등)를 모두 공백으로 바꿔주어 뒤따를 형태소 분석기가 더 수월하게 작업할 수 있도록 도와주는 전처리 과정입니다.

#### **3. 응용 가능한 예제**

**"유튜브 댓글 데이터에서 이모티콘 및 특수문자 제거하기"**

유튜브 API로 댓글 데이터를 수집하면 다양한 이모티콘과 특수문자가 섞여 있습니다. 감성 분석 등을 수행하기 전에 위와 동일한 `re.sub` 로직을 사용하여 한글, 영어, 숫자만 남기고 텍스트를 깨끗하게 정제하는 데 활용할 수 있습니다.

---

### **Part 3: 한글 형태소 분석 (핵심 명사 추출)**

본격적으로 `KoNLPy`를 사용하여 정제된 텍스트에서 의미 있는 단어인 '명사'를 추출하는, 한글 자연어 처리의 핵심 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# konlpy 라이브러리의 tag 모듈에서 Okt 형태소 분석기를 가져옴
from konlpy.tag import Okt
from collections import Counter

# Okt 형태소 분석기 객체를 생성
nlp = Okt()

# nlp.nouns(message): message 전체 텍스트에서 '명사'에 해당하는 단어만 추출하여 리스트로 반환
message_N = nlp.nouns(message)

# Counter 클래스를 이용해 명사 리스트(message_N)에 있는 각 단어의 출현 횟수를 계산
# 결과: {'부산': 150, '맛집': 145, '서면': 120, ...} 과 같은 딕셔너리 형태
count = Counter(message_N)

# 빈도가 높은 상위 80개 단어 중, 글자 길이가 2 이상인 단어만 필터링하여 저장
word_count = dict()
# count.most_common(80): 빈도수 상위 80개의 (단어, 빈도수) 튜플 리스트를 반환
for tag, counts in count.most_common(80):
    # '곳', '것', '수' 등 의미가 적은 한 글자 명사를 제외하기 위한 조건
    if (len(str(tag)) > 1):
        word_count[tag] = counts
```

#### **2. 해당 설명**

이 부분이 바로 ch2-1(영어 분석)과의 가장 큰 차이점입니다. 영어에서는 `word_tokenize`로 단순히 쪼개기만 했지만, 여기서는 `Okt` 형태소 분석기(`nlp`)의 `nouns()` 메소드를 사용합니다. 이 메소드는 `Okt`가 가진 한국어 사과 문법 지식을 바탕으로, 긴 `message` 문자열을 분석하여 품사가 **명사(Noun)**인 단어들만 정확하게 골라내 리스트로 만들어 줍니다. 이 과정을 통해 '맛집은'이 아닌 '맛집'이, '음식을'이 아닌 '음식'이 추출되어 정확한 빈도 분석이 가능해집니다. 마지막으로, 분석 결과의 품질을 높이기 위해 `len(str(tag)) > 1` 조건을 추가하여 '것', '곳', '수' 와 같이 자주 등장하지만 큰 의미가 없는 한 글자 명사를 제거하는 필터링 작업도 수행합니다.

#### **3. 추가하고 싶은 내용 (다양한 품사 태깅 기능)**

`Okt`는 명사 추출 외에도 다양한 기능을 제공합니다.
*   `nlp.morphs(text)`: 텍스트를 모든 형태소 단위로 분리. `['부산', '맛집', '은', '정말', '맛있다']`
*   `nlp.pos(text)`: 각 형태소를 품사와 함께 튜플 형태로 분리. `[('부산', 'Noun'), ('맛집', 'Noun'), ('은', 'Josa'), ('정말', 'Adverb'), ('맛있다', 'Adjective')]`

분석 목적에 따라 `pos()`를 사용해 명사(Noun), 동사(Verb), 형용사(Adjective)를 함께 추출하면 더 풍부한 의미 분석이 가능합니다.

---

### **Part 4: 시각화 (한글 폰트 처리 포함)**

분석 결과를 막대 그래프와 워드 클라우드로 시각화하여 인사이트를 도출하는 마지막 단계입니다. 특히, 한글이 깨지지 않도록 폰트를 설정하는 것이 핵심입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

# --- 시각화 공통: 한글 폰트 설정 ---
# 사용할 한글 폰트 파일의 경로를 지정 (Windows의 '맑은 고딕 볼드'체 예시)
font_path = "c:/Windows/Fonts/malgunbd.ttf"
# 폰트 경로를 이용해 폰트 이름을 가져옴
font_name = font_manager.FontProperties(fname=font_path).get_name()
# matplotlib의 기본 폰트를 위에서 지정한 폰트로 설정
matplotlib.rc("font", family=font_name)

# --- 시각화 1: 막대 그래프 ---
plt.figure(figsize=(12, 6))  # 그래프 크기 설정
# ... (xlabel, ylabel, grid 등 설정) ...
sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)
plt.bar(range(len(word_count)), sorted_Values, align="center")
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation="vertical")
plt.show()

# --- 시각화 2: 워드 클라우드 ---
alice_mask = np.array(Image.open("convert_test1.png"))

# WordCloud 객체 생성 시, 한글 폰트 경로를 반드시 지정해주어야 함
wc = WordCloud(font_path=font_path, background_color="white", max_words=1000, mask=alice_mask)

# 단어 빈도 딕셔너리(word_count)로 워드 클라우드 생성
cloud = wc.generate_from_frequencies(word_count)

plt.imshow(cloud) # 생성된 이미지 표시
plt.axis("off")   # 축 정보 숨기기
plt.show()

# 생성된 워드 클라우드를 이미지 파일로 저장
# cloud.to_file(inputFileName + "_cloud.jpg")
```

#### **2. 해당 설명**

시각화 로직 자체는 영어 분석(ch2-1)과 거의 동일하지만, **한글 폰트 처리**라는 결정적인 차이가 있습니다. `matplotlib`과 `wordcloud` 라이브러리는 기본적으로 한글 폰트를 지원하지 않기 때문에, 명시적으로 한글을 지원하는 폰트 파일의 경로(`font_path`)를 지정해주지 않으면 모든 한글이 깨져서 네모(□)로 표시됩니다.

*   **Matplotlib**: `matplotlib.rc("font", family=font_name)` 코드를 통해 라이브러리 전역의 기본 폰트를 한글 폰트로 설정합니다.
*   **WordCloud**: 객체를 생성할 때 `WordCloud(font_path=font_path, ...)`와 같이 `font_path` 인자를 직접 전달해야 합니다.

이 두 가지 핵심적인 폰트 설정을 통해, '부산', '맛집', '서면' 등의 키워드가 막대 그래프와 워드 클라우드에 정상적으로 표시되도록 만들 수 있습니다.