# from collections import Counter
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import nltk
# from text_manager import simplify_purpose, nlp
# import re

# nltk.download('punkt')
# nltk.download('punkt_tab')

# model_dir = "lcw99/t5-base-korean-text-summary"
# # model_dir = "noahkim/KoT5_news_summarization"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# max_input_length = 1024


# def replace_name(name, text):
#     """
#     텍스트에서 주어진 이름을 풀네임으로 교체
#     성씨 뒤에 직책이나 직위가 있을 경우 해당 직위도 함께 교체
#     """
#     # 의존 구문 분석
#     doc = nlp(text)

#     # 풀네임 (여기서는 주어진 이름을 사용할 것)
#     full_name = name

#     # 직책, 직위 단어 목록 (이후 개선 가능)
#     position_words = [
#         "대통령", "국회의원", "국회위원", "최고위원", "당대표", "대변인", "부대표", "비례대표", "원내대표", "전 대표",
#         "전 의원", "당수", "시의원", "지방의회 의원", "장관", "부장관", "청와대 비서실장", "청와대 대변인", "통일부 장관",
#         "경제부총리", "인사혁신처장", "외교부 장관", "법무부 장관", "교육부 장관", "노동부 장관", "사회복지부 장관",
#         "지방자치단체장", "행정자치부 장관", "선거관리위원회 위원", "정치인", "정부 고위 관계자", "정당 대표", "정당 최고위원",
#         "정당 대변인", "정당 부대표",
#     ]

#     # 텍스트에서 각 단어를 분석
#     for sentence in doc.sentences:
#         for i, word in enumerate(sentence.words):
#             # 성씨가 나오고 직책/직위가 뒤따를 경우 풀네임으로 교체
#             if word.text == name[0] and i + 1 < len(sentence.words) and any(pos_word in sentence.words[i+1].text for pos_word in position_words):
#                 # 풀네임으로 교체
#                 word.text = full_name
#                 print(f"교체된 이름: {full_name}")

#     # 수정된 텍스트 반환
#     summary = " ".join([word.text for word in sentence.words])
#     summary = re.sub(r'\s+\.', '.', string=summary)

#     return summary


# # def remove_repeated_patterns(text):
# #     words = text.split()  # 단어 단위로 분리
# #     n = len(words)

# #     # 반복 패턴을 저장할 변수
# #     result = []

# #     for i in range(n):
# #         found = False
# #         # 슬라이딩 윈도우로 앞뒤 구간 비교
# #         for j in range(i+1, n+1):
# #             pattern = " ".join(words[i:j])  # 현재 슬라이딩 윈도우에서의 패턴
# #             if pattern in " ".join(words[j:]):  # 뒤쪽에 같은 패턴이 있으면
# #                 found = True
# #                 print("pattern!: ", pattern)
# #                 break  # 반복되는 패턴이 있으면 더 이상 진행하지 않고 break
# #         if not found:
# #             result.append(words[i])  # 중복이 아닌 경우 결과에 추가

# #     return " ".join(result)  # 리스트를 다시 문자열로 변환하여 반환

# # def remove_repeated_patterns(text):
# #     words = text.split()  # 단어 단위로 분리
# #     n = len(words)

# #     # 반복 패턴을 저장할 변수
# #     result = []

# #     i = 0
# #     while i < n:
# #         pattern_found = False
# #         # 슬라이딩 윈도우로 앞뒤 구간 비교
# #         for j in range(i + 1, n + 1):
# #             pattern = " ".join(words[i:j])  # 현재 슬라이딩 윈도우에서의 패턴
# #             # 뒤쪽에 같은 패턴이 있으면
# #             if pattern in " ".join(words[j:]):
# #                 pattern_found = True
# #                 print("Found repeated pattern: ", pattern)
# #                 # j까지의 단어들은 중복되므로 건너뜁니다
# #                 i = j
# #                 break

# #         if not pattern_found:
# #             result.append(words[i])  # 중복되지 않은 경우 결과에 추가
# #             i += 1  # 다음 단어로 이동

# #     return " ".join(result)  # 리스트를 다시 문자열로 변환하여 반환

# def remove_repeated_patterns(text):
#     words = text.split()  # 단어 단위로 분리
#     n = len(words)

#     # 반복 패턴을 저장할 변수
#     result = []
#     deleted_patterns = []  # 삭제된 패턴을 기록할 리스트

#     # 뒤에서부터 슬라이딩 윈도우로 중복을 찾기
#     i = 0
#     while i < n:
#         found = False

#         # 슬라이딩 윈도우 사이즈를 큰 수에서 작은 수로 줄여가며 체크
#         for window_size in reversed(range(2, n + 1)):  # 최소 윈도우 사이즈는 2부터 시작
#             for j in range(i + window_size, n + 1):
#                 pattern = " ".join(words[i:j])  # 현재 슬라이딩 윈도우에서의 패턴
#                 if pattern in " ".join(words[j:]):  # 뒤에 있는 동일한 패턴 찾기
#                     deleted_patterns.append(pattern)  # 삭제된 패턴 기록
#                     found = True
#                     break  # 반복되는 패턴이 있으면 더 이상 진행하지 않고 break

#             if found:
#                 break  # 큰 윈도우에서 패턴을 찾으면 더 이상 작은 윈도우로 진행하지 않음

#         # 중복된 문장 처리 후, 현재 단어를 결과에 추가
#         if i < len(words):  # i가 words 리스트 내에서 범위를 벗어나지 않도록 처리
#             result.append(words[i])

#         # 패턴이 발견된 경우 그 이후 단어를 처리하므로 i를 그 위치로 이동
#         if found:
#             i = j  # 패턴을 제거한 후, i를 j로 이동하여 그 이후부터 처리
#         else:
#             i += 1  # 중복이 없으면 다음 단어로 이동


#     print("Processed Text: ", " ".join(result))
#     print("Deleted Patterns: ", deleted_patterns)

#     # 결과 문자열과 삭제된 패턴을 반환
#     return " ".join(result)


# def get_summary(title, body, name, max_length):
#     # inputs = ["extract keyword: " + "title: " +
#     #           title + "speaker: " + name + "body: " + body]
#     inputs = ["sammary: " + title + body]
#     # print(inputs)

#     inputs = tokenizer(inputs, max_length=max_input_length,
#                        truncation=True, return_tensors="pt")
#     # output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=1, max_length=30)
#     output = model.generate(**inputs, num_beams=16,
#                             do_sample=False, min_length=1, max_length=max_length)
#     decoded_output = tokenizer.batch_decode(
#         output, skip_special_tokens=True)[0]
#     topic = nltk.sent_tokenize(decoded_output.strip())[0]

#     return topic


# def extract_topic(title, body, name, max_length=60):
#     topic = get_summary(title, body, name, max_length)
#     print("요약 1: ", topic)
#     topic = replace_name(name=name, text=topic)
#     print("요약 1 풀네임 교체: ", topic)
#     topic = remove_repeated_patterns(topic)
#     print("요약 1 중복 제거: ", topic)
#     # topic = get_summary("", topic, "", 50)
#     # print("요약 2: ", topic)
#     # topic = replace_name(name=name, text=topic)
#     # print("요약 2 풀네임 교체: ", topic)
#     topic = simplify_purpose(sentence=topic, name=name)
#     print("요약 2 개조식: ", topic)

#     return topic


# if __name__ == "__main__":
#     title = """
# 친명계가 꺼낸 ‘투표 불참 방탄’… 한동훈 “더 저질 방탄”
#     """
#     body = """
# 고민정 최고위원은 CBS라디오 인터뷰에서 "김은경 혁신위에서 제안했던 체포동의안에 대한 민주당의 스탠스, 그리고 거기에 대한 지도부의 답변은 있었던 상황"이라며 "그 말(불체포특권 포기)을 번복하자는 의미냐"고 따졌다

#     """
#     name = "고민정"
#     topic = extract_topic(title, body, name, max_length=60)


# import re
# from text_manager import nlp

# def extract_nouns(text):
#     # 텍스트 분석
#     doc = nlp(text)
#     result = []
#     for sent in doc.sentences:
#         for word in sent.words:
#             for (lemma, xpos) in zip(word.lemma.split("+"), word.xpos.split("+")):
#                 if xpos.startswith('n'):
#                     result.append(lemma)
#     print("명사 추출: ", " ".join(result))
#     return result


# def find_common_words(text1, text2):
#     # 두 텍스트를 단어 단위로 분리
#     words1 = extract_nouns(text1)
#     words2 = extract_nouns(text2)

#     # 두 텍스트에서 공통된 단어를 찾기
#     common_words_set = set(words1) & set(words2)

#     # 공통 단어들이 원본 순서를 유지하며 중복 없이 나타나도록 필터링
#     common_words = [word for word in words1 if word in common_words_set]

#     print("공통 단어: ", " ".join(common_words))

#     return common_words


# def find_word_with_context(word, sentence, n=2):
#     """
#     주어진 단어를 문장에서 찾아 주변 단어와 함께 묶어서 반환하는 함수
#     :param word: 찾을 단어
#     :param sentence: 단어를 찾을 문장
#     :param n: 주변 단어의 범위 (기본값은 2, 즉 앞뒤 2개의 단어)
#     :return: 해당 단어와 주변 단어를 포함하는 문맥
#     """
#     # 문장을 단어로 분리
#     words = sentence.split()

#     # 단어가 문장에 있는지 확인하고 해당 단어의 인덱스 찾기
#     for i, w in enumerate(words):
#         if word in w:  # 부분 문자열로 포함되는지 확인
#             # 문맥 범위 설정: 앞뒤로 n개 단어를 포함
#             start = max(0, i - n)  # 시작 인덱스 (음수가 되지 않도록 처리)
#             end = min(len(words), i + n + 1)  # 끝 인덱스 (문장 길이를 넘지 않도록 처리)

#             # 해당 단어와 주변 단어들을 포함한 문맥을 반환
#             context = words[start:end]
#             return " ".join(context)
#     return None


# def extract_topic(title, body, purpose, name, prev_topic):
#     # 중복된 코드를 함수로 추출
#     def get_result(common_words, reference_text):
#         if len(common_words) == 1:
#             return find_word_with_context(common_words[0], reference_text)
#         else:
#             return " ".join(common_words)

#     # case 1: title과 body에서 공통된 단어 추출
#     title_body = find_common_words(title, body)
#     if title_body:
#         return get_result(title_body, title)

#     # case 2: title과 purpose에서 공통된 단어 추출
#     title_purpose = find_common_words(title, purpose)
#     if title_purpose:
#         return get_result(title_purpose, title)

#     # case 3: body와 purpose에서 공통된 단어 추출
#     body_purpose = find_common_words(body, purpose)
#     if body_purpose:
#         return get_result(body_purpose, purpose)

#     # case 4: 이전 주제 반환
#     if prev_topic:
#         return prev_topic
#     else:
#         return title

# from text_manager import nlp


# def extract_nouns(text):
#     # 텍스트 분석
#     doc = nlp(text)
#     # print(doc)
#     result = []
#     for sent in doc.sentences:
#         for word in sent.words:
#             if word.upos in ["ADV", "ADJ", "VERB"]:
#                 for (lemma, xpos) in zip(word.lemma.split("+"), word.xpos.split("+")):
#                     if xpos.startswith('n'):
#                         result.append(lemma)
#             else:
#                 for (lemma, xpos) in zip(word.lemma.split("+"), word.xpos.split("+")):
#                     if xpos.startswith('n'):
#                         result.append(lemma)
#         print("명사 추출: ", " ".join(result))
#     return result


# def find_common_words(text1, text2):
#     # 두 텍스트를 단어 단위로 분리
#     words1 = extract_nouns(text1)
#     words2 = extract_nouns(text2)

#     # 두 텍스트에서 공통된 단어를 찾기
#     common_words_set = set(words1) & set(words2)

#     # 공통 단어들이 원본 순서를 유지하며 중복 없이 나타나도록 필터링
#     common_words = [word for word in words1 if word in common_words_set]

#     print("공통 단어: ", " ".join(common_words))

#     return common_words


# def find_word_with_context(word, sentence, n=2):
#     """
#     주어진 단어를 문장에서 찾아 주변 단어와 함께 묶어서 반환하는 함수
#     :param word: 찾을 단어
#     :param sentence: 단어를 찾을 문장
#     :param n: 주변 단어의 범위 (기본값은 2, 즉 앞뒤 2개의 단어)
#     :return: 해당 단어와 주변 단어를 포함하는 문맥
#     """
#     # 문장을 단어로 분리
#     words = sentence.split()

#     # 단어가 문장에 있는지 확인하고 해당 단어의 인덱스 찾기
#     for i, w in enumerate(words):
#         if word in w:  # 부분 문자열로 포함되는지 확인
#             # 문맥 범위 설정: 앞뒤로 n개 단어를 포함
#             start = max(0, i - n)  # 시작 인덱스 (음수가 되지 않도록 처리)
#             end = min(len(words), i + n + 1)  # 끝 인덱스 (문장 길이를 넘지 않도록 처리)

#             # 해당 단어와 주변 단어들을 포함한 문맥을 반환
#             context = words[start:end]
#             return " ".join(context)
#     return None


# def find_related_words(sentence, target_word):
#     doc = nlp(sentence)  # 문장을 분석하여 doc 객체 생성

#     # 문장에서 단어들을 확인
#     result = []
#     for sent in doc.sentences:
#         # target_word를 포함하는 토큰을 찾기
#         for i, token in enumerate(sent.words):
#             # if token.text == target_word:
#             if target_word in token.text:
#                 # 자기 자신도 포함
#                 result.append(token.text)
#                 print("appended self")

#                 # 부모 단어 추출
#                 if token.head != 0:  # head가 0이면 부모가 없다는 의미
#                     parent = sent.words[token.head - 1]  # 부모 토큰 얻기 (인덱스 - 1)
#                     result.append(parent.text)

#                 # 자식 단어 추출 (deps가 None이 아닐 때만)
#                 if token.deps is not None:
#                     # 자식 토큰들을 추가
#                     result.extend(
#                         [sent.words[dep[0] - 1].text for dep in token.deps])

#                 # 만약 None이거나 연관된 단어가 없다면 주변 단어와 target_word를 묶어 반환
#                 if len(result) == 1:
#                     # 앞뒤 단어가 존재하면 그 단어와 함께 묶어서 반환
#                     if i > 0:  # 앞 단어 존재
#                         result.append(sent.words[i-1].text)
#                     result.append(token.text)
#                     if i < len(sent.words) - 1:  # 뒤 단어 존재
#                         result.append(sent.words[i+1].text)

#     # 결과는 원본 순서대로 정렬된 상태로 반환
#     return " ".join(result)


# def extract_topic(title, body, purpose, name, prev_topic):
#     # 중복된 코드를 함수로 추출
#     def get_result(common_words, reference_text):
#         if len(common_words) == 1:
#             # return find_word_with_context(common_words[0], reference_text)
#             return find_related_words(reference_text, common_words[0])
#         else:
#             return " ".join(common_words)

#     # case 1: title과 body에서 공통된 단어 추출
#     title_body = find_common_words(title, body)
#     if title_body:
#         return get_result(title_body, title)

#     # case 2: title과 purpose에서 공통된 단어 추출
#     title_purpose = find_common_words(title, purpose)
#     if title_purpose:
#         return get_result(title_purpose, title)

#     # case 3: body와 purpose에서 공통된 단어 추출
#     body_purpose = find_common_words(body, purpose)
#     if body_purpose:
#         return get_result(body_purpose, purpose)

#     # case 4: 이전 주제 반환
#     if prev_topic:
#         return prev_topic
#     else:
#         return title

from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from text_manager import simplify_purpose, nlp
import re

nltk.download('punkt')
nltk.download('punkt_tab')

model_dir = "lcw99/t5-base-korean-text-summary"
# model_dir = "noahkim/KoT5_news_summarization"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 1024


def get_summary(title, body = "", name = "", max_length = 30):
    inputs = ["sammary: " + title]

    inputs = tokenizer(inputs, max_length=max_input_length,
                       truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=16,
                            do_sample=False, min_length=1, max_length=max_length)
    decoded_output = tokenizer.batch_decode(
        output, skip_special_tokens=True)[0]
    topic = nltk.sent_tokenize(decoded_output.strip())[0]

    return topic


def replace_name(name, text):
    """
    텍스트에서 주어진 이름을 풀네임으로 교체
    성씨 뒤에 직책이나 직위가 있을 경우 해당 직위도 함께 교체
    """
    # 의존 구문 분석
    doc = nlp(text)

    # 풀네임 (여기서는 주어진 이름을 사용할 것)
    full_name = name

    # 직책, 직위 단어 목록 (이후 개선 가능)
    position_words = [
        "대통령", "국회의원", "국회위원", "최고위원", "당대표", "대변인", "부대표", "비례대표", "원내대표", "전 대표",
        "전 의원", "당수", "시의원", "지방의회 의원", "장관", "부장관", "청와대 비서실장", "청와대 대변인", "통일부 장관",
        "경제부총리", "인사혁신처장", "외교부 장관", "법무부 장관", "교육부 장관", "노동부 장관", "사회복지부 장관",
        "지방자치단체장", "행정자치부 장관", "선거관리위원회 위원", "정치인", "정부 고위 관계자", "정당 대표", "정당 최고위원",
        "정당 대변인", "정당 부대표",
    ]

    # 텍스트에서 각 단어를 분석
    for sentence in doc.sentences:
        for i, word in enumerate(sentence.words):
            # 성씨가 나오고 직책/직위가 뒤따를 경우 풀네임으로 교체
            if word.text == name[0] and i + 1 < len(sentence.words) and any(pos_word in sentence.words[i+1].text for pos_word in position_words):
                # 풀네임으로 교체
                word.text = full_name
                print(f"교체된 이름: {full_name}")

    # 수정된 텍스트 반환
    summary = " ".join([word.text for word in sentence.words])
    summary = re.sub(r'\s+\.', '.', string=summary)

    return summary

def remove_repeated_patterns(text):
    words = text.split()  # 단어 단위로 분리
    n = len(words)

    # 반복 패턴을 저장할 변수
    result = []
    deleted_patterns = []  # 삭제된 패턴을 기록할 리스트

    # 뒤에서부터 슬라이딩 윈도우로 중복을 찾기
    i = 0
    while i < n:
        found = False

        # 슬라이딩 윈도우 사이즈를 큰 수에서 작은 수로 줄여가며 체크
        for window_size in reversed(range(2, n + 1)):  # 최소 윈도우 사이즈는 2부터 시작
            for j in range(i + window_size, n + 1):
                pattern = " ".join(words[i:j])  # 현재 슬라이딩 윈도우에서의 패턴
                if pattern in " ".join(words[j:]):  # 뒤에 있는 동일한 패턴 찾기
                    deleted_patterns.append(pattern)  # 삭제된 패턴 기록
                    found = True
                    break  # 반복되는 패턴이 있으면 더 이상 진행하지 않고 break

            if found:
                break  # 큰 윈도우에서 패턴을 찾으면 더 이상 작은 윈도우로 진행하지 않음

        # 중복된 문장 처리 후, 현재 단어를 결과에 추가
        if i < len(words):  # i가 words 리스트 내에서 범위를 벗어나지 않도록 처리
            result.append(words[i])

        # 패턴이 발견된 경우 그 이후 단어를 처리하므로 i를 그 위치로 이동
        if found:
            i = j  # 패턴을 제거한 후, i를 j로 이동하여 그 이후부터 처리
        else:
            i += 1  # 중복이 없으면 다음 단어로 이동


    # print("Processed Text: ", " ".join(result))
    # print("Deleted Patterns: ", deleted_patterns)

    # 결과 문자열과 삭제된 패턴을 반환
    return " ".join(result)

def extract_nouns(text):
    # 텍스트 분석
    doc = nlp(text)
    # print(doc)
    result = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ["ADV", "ADJ", "VERB"]:
                for (lemma, xpos) in zip(word.lemma.split("+"), word.xpos.split("+")):
                    if xpos.startswith('n'):
                        result.append(lemma)
            else:
                for (lemma, xpos) in zip(word.lemma.split("+"), word.xpos.split("+")):
                    if xpos.startswith('n'):
                        result.append(lemma)
        print("명사 추출: ", " ".join(result))
    return result


def find_common_words(text1, text2):
    # 두 텍스트를 단어 단위로 분리
    words1 = extract_nouns(text1)
    words2 = extract_nouns(text2)

    # 두 텍스트에서 공통된 단어를 찾기
    common_words_set = set(words1) & set(words2)

    # 공통 단어들이 원본 순서를 유지하며 중복 없이 나타나도록 필터링
    common_words = [word for word in words1 if word in common_words_set]

    print("공통 단어: ", " ".join(common_words))

    return common_words


def find_word_with_context(word, sentence, n=2):
    """
    주어진 단어를 문장에서 찾아 주변 단어와 함께 묶어서 반환하는 함수
    :param word: 찾을 단어
    :param sentence: 단어를 찾을 문장
    :param n: 주변 단어의 범위 (기본값은 2, 즉 앞뒤 2개의 단어)
    :return: 해당 단어와 주변 단어를 포함하는 문맥
    """
    # 문장을 단어로 분리
    words = sentence.split()

    # 단어가 문장에 있는지 확인하고 해당 단어의 인덱스 찾기
    for i, w in enumerate(words):
        if word in w:  # 부분 문자열로 포함되는지 확인
            # 문맥 범위 설정: 앞뒤로 n개 단어를 포함
            start = max(0, i - n)  # 시작 인덱스 (음수가 되지 않도록 처리)
            end = min(len(words), i + n + 1)  # 끝 인덱스 (문장 길이를 넘지 않도록 처리)

            # 해당 단어와 주변 단어들을 포함한 문맥을 반환
            context = words[start:end]
            return " ".join(context)
    return None


def find_related_words(sentence, target_word):
    doc = nlp(sentence)  # 문장을 분석하여 doc 객체 생성

    # 문장에서 단어들을 확인
    result = []
    for sent in doc.sentences:
        # target_word를 포함하는 토큰을 찾기
        for i, token in enumerate(sent.words):
            # if token.text == target_word:
            if target_word in token.text:
                # 자기 자신도 포함
                result.append(token.text)
                print("appended self")

                # 부모 단어 추출
                if token.head != 0:  # head가 0이면 부모가 없다는 의미
                    parent = sent.words[token.head - 1]  # 부모 토큰 얻기 (인덱스 - 1)
                    result.append(parent.text)

                # 자식 단어 추출 (deps가 None이 아닐 때만)
                if token.deps is not None:
                    # 자식 토큰들을 추가
                    result.extend(
                        [sent.words[dep[0] - 1].text for dep in token.deps])

                # 만약 None이거나 연관된 단어가 없다면 주변 단어와 target_word를 묶어 반환
                if len(result) == 1:
                    # 앞뒤 단어가 존재하면 그 단어와 함께 묶어서 반환
                    if i > 0:  # 앞 단어 존재
                        result.append(sent.words[i-1].text)
                    result.append(token.text)
                    if i < len(sent.words) - 1:  # 뒤 단어 존재
                        result.append(sent.words[i+1].text)

    # 결과는 원본 순서대로 정렬된 상태로 반환
    return " ".join(result)


def extract_topic(title, body, purpose, name, prev_topic):
    # 중복된 코드를 함수로 추출
    def get_result(common_words, reference_text):
        if len(common_words) == 1:
            # return find_word_with_context(common_words[0], reference_text)
            return find_related_words(reference_text, common_words[0])
        else:
            return " ".join(common_words)

    # case 1: title과 body에서 공통된 단어 추출
    title_body = find_common_words(title, body)
    if title_body:
        return get_result(title_body, title)

    # case 2: title과 purpose에서 공통된 단어 추출
    title_purpose = find_common_words(title, purpose)
    if title_purpose:
        return get_result(title_purpose, title)

    # case 3: body와 purpose에서 공통된 단어 추출
    body_purpose = find_common_words(body, purpose)
    if body_purpose:
        return get_result(body_purpose, purpose)

    # case 4: 이전 주제 반환
    if prev_topic:
        return prev_topic
    else:
        topic = get_summary(title)
        topic = replace_name(topic, name)
        topic = remove_repeated_patterns(topic)
        topic = simplify_purpose(topic, name)
        return topic
        
        


if __name__ == "__main__":
    title = "고민정 최고위원, 체포동의안 부결 주장"
    body = "고민정 최고위원은 CBS 라디오에서 체포동의안 부결 주장에 대해 언급했다."

    purpose = "정치적 발언에 대한 논란"
    name = "고민정"
    prev_topic = "정치적 발언의 중요성"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)

    title = "국회, 부패 척결에 나서다"
    body = "국회는 부패 척결을 위해 노력하고 있다."
    purpose = "부패와 정치 개혁에 대한 의견"

    name = "이상민"
    prev_topic = "정치개혁"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)

    title = "한미 정상 회담"
    body = "한미 정상 회담에서 양국의 협력 방안이 논의되었다."
    purpose = "외교 관계에서의 협력 방안"

    name = "박진"
    prev_topic = "국제 외교"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)

    title = "경제 성장률 예측"
    body = "올해 경제 성장률은 예상보다 낮았다."
    purpose = "경제 정책 개선"

    name = "이재명"
    prev_topic = "경제 문제의 해결책"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)



if __name__ == "__main__":
    title = "고민정 최고위원, 체포동의안 부결 주장"
    body = "고민정 최고위원은 CBS 라디오에서 체포동의안 부결 주장에 대해 언급했다."

    purpose = "정치적 발언에 대한 논란"
    name = "고민정"
    prev_topic = "정치적 발언의 중요성"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)

    title = "국회, 부패척결에 나서다"
    body = "국회는 부패 척결을 위해 노력하고 있다."
    purpose = "부패와 정치 개혁에 대한 의견"

    name = "이상민"
    prev_topic = "정치개혁"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)

    title = "한미 정상 회담"
    body = "한미 정상 회담에서 양국의 협력 방안이 논의되었다."
    purpose = "외교 관계에서의 협력 방안"

    name = "박진"
    prev_topic = "국제 외교"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)

    title = "경제 성장률 예측"
    body = "올해 경제 성장률은 예상보다 낮았다."
    purpose = "경제 정책 개선"

    name = "이재명"
    prev_topic = "경제 문제의 해결책"

    # 함수 호출
    result = extract_topic(title, body, purpose, name, prev_topic)
    print(result)
