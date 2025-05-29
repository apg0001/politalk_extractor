from text_manager import *
import stanza
import re
import torch

# Stanza 모델 로드
# stanza.download('ko')  # 한국어 모델 다운로드
# GPU 사용 가능 시 GPU로 처리
# nlp = stanza.Pipeline('ko', use_gpu=torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 중요한 정보
important_deprels = {"nsubj",  # 주어
                     "isubj",  # 간접 주어
                     #  "dislocated",  # 이동된 주어
                     "root",  # 루트
                     "obj",  # 목적어
                     "iobj",  # 간접 목적어
                     "obl",  # oblique object
                     "advcl",  # adverbial clause
                     "conj",  # 연결어
                     "compound",  # 복합어
                     "fixed",  # 고정어
                     "nmod",  # 관계어(명사 수식어, Nominal modifier)
                     "acl",  # 관계절(부사절적 수식어, Adnominal clause)
                     "aux",  # 조동사(Auxiliary)
                     }  # 중요한 관계들

important_xpos = {"ncn+jca+jxt",
                  }
# 불필요한 정보
unimportant_xpos = {"pvg+ecc",  # ~라며 등
                    "pvg+ef+jcr",  # ~고 등
                    "pvg+ecx+jcr",  # ~고 등
                    "ncn+ef+jcr",  # ~이라고 등
                    "pvg+ecx",  # ~라며 등
                    "ncn+jxc",  # ~면서도 등
                    }  # 불필요한 품사


def extract_and_clean_quotes(text):
    """
    텍스트에서 쌍따옴표로 묶인 문장을 추출하고 원래 문단에서 제거

    Args:
        text (str): 입력 텍스트

    Returns:
        tuple: (추출된 쌍따옴표 문장 리스트, 쌍따옴표 문장이 제거된 원래 문단)
    """

    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    text = text.replace("‘", "\'")
    text = text.replace("'", "\'")
    text = text.replace("’", " \'")
    quotes = re.findall(r'"(.*?)"', text)

    # 쌍따옴표 안 내용 + 바로 뒤에 붙은 한 단어(띄어쓰기 포함해서 최대 2글자 정도)까지 제거
    pattern = r'"[^"]*"(?:\s*\S+)?'

    cleaned_text = re.sub(pattern, '', text)
    return quotes, cleaned_text


def preprocess_text(text):
    """
    텍스트 전처리: 특수문자 제거 및 공백 정리

    Args:
        text (str): 입력 텍스트

    Returns:
        str: 전처리된 텍스트
    """
    text = re.sub(r'\([^)]*\)', '', text)  # 괄호 내용 제거
    text = re.sub(r'[^\w\s.!?]', '', text)   # 특수문자 제거, 문장 분할용 구두점은 유지
    text = re.sub(r'\s+', ' ', text)      # 다중 공백 정리
    return text.strip()


def get_summary(name, text):
    """
    텍스트를 요약
    """
    # 의존 구문 분석
    doc = nlp(text)
    # print(doc)
    filtered_words = []
    quote_flag = False
    # print(text + "\n")

    for sentence in doc.sentences:
        # print(sentence)
        size = 0
        try:
            size = len(doc.sentence.words)
            # print(size)
        except:
            size = 0
        for word in sentence.words:
            # print(f"{word.text} : {word.deprel}")
            # 중요한 관계에 해당하는 단어만 남김
            if (word.deprel in important_deprels) or \
               (word.xpos in important_xpos) or \
               (word.deprel == "acl" and sentence.words[word.id].deprel == "obl") or \
               (word.id >= 2 and word.deprel == "dislocated" and word.xpos == "ncn+ncn+jxt" and sentence.words[word.id-2].text == "겸") or \
               (word.xpos == "ncn+jco" and sentence.words[word.id].text == "요구했다") or \
               (word.id >= 2 and word.deprel == "dep" and sentence.words[word.id-2].xpos == "nq"):
                """
                1.
                2.
                3. "~ 겸" 뒤에 나오는 주어 포함(원내대표 등)
                4. "~ 요구했다" 앞에 나오는 단어 포함
                5.
                """
                # 불필요한 품사 필터링
                if (word.xpos in unimportant_xpos) or \
                   (word.text == name) or \
                   (word.id >= 2 and word.xpos in ["ncn+ncn+jxt", "ncn+jtx", "ncn+ncn+jxc"] and sentence.words[word.id-2].text in [name, name[0]]) or \
                   (word.id >= 2 and word.text == "당" and sentence.words[word.id-2].text == "같은") or \
                   (word.id < size - 3 and word.deprel == "compound" and sentence.words[word.id].text in [name, name[0]]) or \
                   (word.id >= 2 and word.deprel == "compound" and sentence.words[word.id-2].text in [name, name[0]]) or \
                   (word.id >= 2 and word.text == "전" and sentence.words[word.id-2].text in [name, name[0]] and sentence.words[word.id].deprel == "dislocated") or \
                   (word.id >= 2 and word.text == "전" and sentence.words[word.id-2].text in [name, name[0]] and sentence.words[word.id].deprel == "advcl"):
                    #    (word.text in [name, name[0]] or sentence.words[word.id-2].xpos in ["ncn+ncn+jxt", "ncn+jtx"]):
                    """
                    1.
                    2.
                    3.
                    4. "같은" 뒤에 "당"이 나오면 제거
                    5. 발언자 이름 앞에 수식어가 나오면 제거
                    6. 발언자 이름 뒤에 수식어가 나오면 제거
                    """
                    _ = None
                    # if (int(word.id) == 2):
                    #     print(word)
                else:
                    filtered_words.append(word.text)
                    # print(
                    #     f"appended : {word.text} -> {' '.join(filtered_words)}")
            # 작은 따옴표 안에 있는 내용은 살리기
            elif word.deprel == "punct" and word.text == "\'" and quote_flag == False:
                filtered_words.append(word.text)
                quote_flag = True
                # print("start quote : " + word.text)
            elif word.deprel == "punct" and word.text == "\'" and quote_flag == True:
                filtered_words.append(word.text)
                quote_flag = False
                # print("end quote : " + word.text)
            elif quote_flag == True:
                filtered_words.append(word.text)
                # print("in quote : " + word.text)

    # 단어를 조합해 요약 문장 생성
    summary = " ".join(filtered_words)
    # summary = preprocess_text(summary)
    summary = normalize_spaces_inside_single_quotes(summary)
    # 요약 반환
    return summary


def extract_purpose(name, title, body1, body2, prev):
    # '이어', '이어서', '그러면서', '그는' 등의 순접으로 이루어져 있다면 이전 행과 같은 목적, 배경, 취지로 고려
    # if find_sequential_conjunction(body1) is not None:
    #     return prev

    # 1. 문장을 마침표 기준으로 나눔
    # 2. 문장에서 쌍따옴표 문장 추출 및 제거
    # 3. 이름(full name) 또는 성(last name)이 포함된 경우만 추출

    # 1. 문장을 마침표 기준으로 나눔
    # sentences_body1 = [sentence.strip()
    #                    for sentence in body1.split(".") if sentence.strip()]

    sentences_body1 = split_preserving_quotes(body1)
    # print(f"sentences_body1: {sentences_body1}")

    # 2. 이름(full name) 또는 성(last name)이 포함된 경우만
    keywords = [name, f"{name[0]} "]
    filtered_body1 = filter_sentences_by_name(sentences_body1, keywords)
    # print(f"filtered_body1: {filtered_body1}")

    # 3. 쌍따옴표 문장 추출
    # extracted_data = [extract_and_clean_quotes(
    #     filtered_sentence_body1) for filtered_sentence_body1 in filtered_body1]
    extracted_data = [extract_and_clean_quotes(
        filtered_sentence_body1) for filtered_sentence_body1 in sentences_body1]

    # 리스트가 비어 있으면 빈 리스트로 초기화
    if extracted_data:
        quotes, sentences_body1_clean = zip(*extracted_data)
        quotes, sentences_body1_clean = list(
            quotes), list(sentences_body1_clean)
    else:
        quotes, sentences_body1_clean = [], []

    filtered_body1_clean = "  ".join(sentences_body1_clean)
    print(f"filtered_body1_clean: {filtered_body1_clean}")

    # 요약문 생성
    summaries_body1 = get_summary(name, filtered_body1_clean)

    print(f"summaries_body1: {summaries_body1}")
    # print(f"simplify_purpose: {simplify_purpose(summaries_body1, name)}")

    return simplify_purpose(summaries_body1, name)
    """
    # 쌍따옴표 문장 추출
    quotes, body1_clean = extract_and_clean_quotes(body1)
    _, body2_clean = extract_and_clean_quotes(body2)

    # 본문 1: 문단을 마침표 기준으로 나눔
    sentences_body1 = [sentence.strip()
                       for sentence in body1_clean.split(".") if sentence.strip()]

    # 이름(full name) 또는 성(last name)이 포함된 경우만
    keywords = [name, f"{name[0]} "]
    filtered_body1 = filter_sentences_by_name(sentences_body1, keywords)
    print(f"filtered_body1: {filtered_body1}")
    
    summaries_body1 = [get_summary(name, sentence)
                       for sentence in filtered_body1]
                       
    return "  ".join([simplify_purpose(summary, name) for summary in summaries_body1])
    """

    # body1에서 추출한 문장의 단어 수가 3개 이하인 경우 body2에서 추출한 문장을 반환
    # if len(body1_summary.split()) <= 3:
    #     body2_summary = get_summary(name, body2_clean)

    #     # body2에서 추출한 문장의 단어 수가 3개 이하인 경우 제목에서 추출한 문장을 반환
    #     # if len(body2_summary.split()) <= 3:
    #     #     title_summary = get_summary(title)
    #     #     return title_summary

    #     # 간소화된 표현으로 반환
    #     return simplify_purpose(body2_summary)

    # 간소화된 표현으로 반환
    # return [simplify_purpose(summary) for summary in filtered_body1]


if __name__ == "__main__":
    # 예제 실행
    name = "김씨"
    title = "최강욱, 이동재 명예훼손 혐의로 또 검찰 송치"
    body1 = '''
김 의원은 이에 "수도권에서만 어필하면 전국 정당이 되느냐, MZ세대만 얻으면 전국 정당이 될 수 있는 거냐"라고 반문하며 "전 국민을 상대로 지지층을 확보하고 전 지역을 상대로 지지층을 확보해야 한다. ‘특정 지역만 지지받으면 된다, 특정 계층만 지지받으면 된다’라는 것은 매우 협소한 의견"이라고 지적했다

'''
    body2 = '''
같은 당 고민정 의원은 "차장이 배석했던 회의가 끝나고 11시54분쯤 02-800-7070으로 (이종섭 전 장관에게) 전화가 가고 그 다음 국방부 장관부터 시작해서 일사천리로 일처리가 된다. 이상하지 않느냐"고 따져물었다
    '''
    prev = "이전 문장입니다."

    # 엔티티 추출
    result = extract_purpose(name, title, body1, body2, prev)
    # print(result)


# todo
# 홑따옴표 안에 있는 내용은 살리자
# 더불어민주당(민주당) 제거

# 완료된 todo

# (했고, )쉼표로 구분(발언자 구분)
# 마이크를 ~고 했고 -> 마이크를 잡고 발언
# 말했지만, 하지만 뒤에 ','컴마 추가해서 강제로 끊기
# 쉼표 다음에 바로 쌍따옴표가 나온 경우 같은 발언자
# ~ 고(라고) 헸다 -> 발언으로 정정
# 검토에 했다 -> 검토에 대해 발언
# "는 내용의" 제거 예외적으로
# "~ 요구"를 수식하는 말이 있는 경우 예외적으로 추가
# 33행 원내대표는 포함
# 발언자를 꾸미는 직함 등 수식어는 전부 제거 ex) 같은 당 (민주당) 고민정 의원 등


# 추가사항

# 결과가 "발언" <- 만 나온 경우
# "발언자"의 발언으로 나오도록 수정
