from text_manager import nlp, filter_sentences_by_name, split_preserving_quotes
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# NLTK 다운로드
nltk.download('punkt')

# KoBART 모델과 토크나이저 로드
model_dir = "lcw99/t5-base-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 1024


def get_summary(text, max_length=40):
    """
    주어진 텍스트에 대해 요약을 생성하는 함수
    :param text: 요약할 텍스트
    :param max_length: 요약문 최대 길이
    :return: 요약된 텍스트
    """
    inputs = [text]
    inputs = tokenizer(inputs, max_length=max_input_length,
                       truncation=True, return_tensors="pt", padding=True)
    output = model.generate(
        **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
    decoded_output = tokenizer.batch_decode(
        output, skip_special_tokens=True)[0]
    topic = nltk.sent_tokenize(decoded_output.strip())[0]
    return topic


def remove_duplicates(text):
    doc = nlp(text)
    lemmas = []
    result = []

    for sentence in doc.sentences:
        for word in sentence.words:
            lemma = word.lemma.split('+')[0]
            if lemma in lemmas:
                continue
            else:
                lemmas.append(lemma)
                result.append(word.text)

    return ' '.join(result)


def replace_name(name, text):
    """
    텍스트에서 주어진 이름을 풀네임으로 교체
    성씨 뒤에 직책이나 직위가 있을 경우 해당 직위도 함께 교체
    """
    full_name = name
    position_words = [
        "대통령", "국회의원", "국회위원", "최고위원", "당대표", "대변인", "부대표", "비례대표", "원내대표", "전 대표",
        "전 의원", "당수", "시의원", "지방의회 의원", "장관", "부장관", "청와대 비서실장", "청와대 대변인", "통일부 장관",
        "경제부총리", "인사혁신처장", "외교부 장관", "법무부 장관", "교육부 장관", "노동부 장관", "사회복지부 장관",
        "지방자치단체장", "행정자치부 장관", "선거관리위원회 위원", "정치인", "정부 고위 관계자", "정당 대표", "정당 최고위원",
        "정당 대변인", "정당 부대표",
    ]
    # 의존 구문 분석
    doc = nlp(text)

    # 텍스트에서 각 단어를 분석하여 직위가 있을 경우 풀네임으로 교체
    for sentence in doc.sentences:
        for i, word in enumerate(sentence.words):
            if word.text == name[0] and i + 1 < len(sentence.words) and any(pos_word in sentence.words[i+1].text for pos_word in position_words):
                word.text = full_name

    # 수정된 텍스트 반환
    summary = " ".join([word.text for word in sentence.words])
    summary = re.sub(r'\s+\.', '.', string=summary)
    return summary


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
    cleaned_text = re.sub(r'"(.*?)"', '', text).strip()
    return quotes, cleaned_text


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
    extracted_data = [extract_and_clean_quotes(
        filtered_sentence_body1) for filtered_sentence_body1 in filtered_body1]

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
    summaries_body1 = get_summary(filtered_body1_clean, 100)
    summaries_body1 = replace_name(name, summaries_body1)
    summaries_body1 = remove_duplicates(summaries_body1)
    print("purpose: ", summaries_body1)

    return summaries_body1


if __name__ == "__main__":
    name = "최강욱"
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
    print(result)
