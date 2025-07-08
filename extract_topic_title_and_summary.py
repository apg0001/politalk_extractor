from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import re
from text_manager import nlp

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


def extract_keyword(text):
    """
    주어진 텍스트에서 명사를 추출하는 함수
    """
    doc = nlp(text)
    # print(doc)
    nouns = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['NOUN', 'VERB', 'ADV', 'ADJ', 'CONJ', 'PROPN', 'DET', 'SCONJ', 'AUX', 'NUM', 'CCONJ']:  # 명사인 경우
                nouns.append(word.text)
    return nouns


def extract_nouns(text):
    """
    주어진 텍스트에서 명사를 추출하는 함수
    """
    doc = nlp(text)
    nouns = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == 'NOUN':  # 명사인 경우
                nouns.append(word.text)
    return nouns


def remove_repeated_nouns(text):
    """
    명사를 추출하여 중복된 명사와 그 뒤에 붙은 조사를 제거하는 함수
    """
    nouns = extract_nouns(text)  # 텍스트에서 명사 추출
    n = len(nouns)

    words = text.split()  # 문장을 단어 단위로 분리
    result = []
    deleted_patterns = []  # 삭제된 패턴 기록할 리스트

    i = 0
    while i < len(words):
        found = False

        # 중복된 명사 찾기
        for j in range(i + 1, len(words)):
            # 패턴을 찾고 뒤에 붙은 조사까지 삭제
            if words[i] == words[j] and words[j] not in deleted_patterns:
                # 중복된 명사와 그 뒤에 있는 조사도 함께 삭제
                deleted_patterns.append(words[j])
                found = True
                break

        if found:
            # 중복된 부분은 result에 추가하지 않음
            i = j  # 패턴이 발견되면 j로 이동
        else:
            result.append(words[i])  # 중복이 없으면 그 단어를 결과에 추가
            i += 1  # 다음 단어로 이동

    return " ".join(result)


def remove_duplicatates(text):
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


def merge_sentences_with_kobart(sentence1, sentence2):
    """
    두 문장을 입력으로 받아서 자연스럽게 결합한 후 요약하는 함수
    :param sentence1: 첫 번째 문장
    :param sentence2: 두 번째 문장
    :return: 결합된 문장
    """
    input_text = sentence1 + " " + sentence2
    inputs = tokenizer([input_text], max_length=1024,
                       return_tensors="pt", truncation=True, padding=True)

    summary_ids = model.generate(
        inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def extract_5w1h(text):
    # 텍스트 분석
    doc = nlp(text)
    # print(doc)

    # 순차적으로 담을 리스트
    result = []

    # 문장 분석
    for sentence in doc.sentences:
        for word in sentence.words:
            # 누가 (주어) - deprel: nsubj, nsubjpass, dislocated
            if word.deprel in ['nsubj', 'nsubjpass', 'dislocated', 'compound'] and word.upos in ['PRON', 'NOUN', 'PROPN']:
                result.append(word.lemma.split('+')[0] if len(word.lemma.split(
                    '+')) == 1 else ''.join(word.lemma.split('+')[:-1]))  # 원형 추출 후 추가
            # 누가 (주어) - dislocated와 compound는 주어를 꾸미는 말이므로 함께 추가
            elif word.deprel in ['dislocated', 'compound'] and word.upos == 'NOUN':
                result.append(word.lemma.split('+')[0] if len(word.lemma.split(
                    '+')) == 1 else ''.join(word.lemma.split('+')[:-1]))  # 원형 추출 후 추가

            # 언제 (시간) - deprel: obl:tmod, compound
            elif word.deprel == 'obl' and 'ncn' in word.xpos:  # 시간 관련
                result.append(word.lemma.split('+')[0])  # 원형 추출 후 추가

            # 어디서 (장소) - deprel: obl:loc, compound
            elif word.deprel == 'obl' and word.upos == 'NOUN':  # 장소 관련
                result.append(word.lemma.split('+')[0])  # 원형 추출 후 추가

            # 무엇을 (목적어) - deprel: obj, iobj, compound
            elif word.deprel in ['obj', 'iobj', 'compound'] and word.xpos == 'ncn':  # 목적어 관련
                result.append(word.lemma.split('+')[0])  # 원형 추출 후 추가

            # 어떻게 (방법) - deprel: obl:manner
            elif word.deprel == 'obl' and word.upos == 'ADV':  # 방법 관련 부사어
                result.append(word.lemma.split('+')[0])  # 원형 추출 후 추가

            # 왜 (이유) - deprel: mark
            elif word.deprel == 'mark' and word.xpos == 'pvg':  # 이유를 나타내는 접속사
                result.append(word.lemma.split('+')[0])  # 원형 추출 후 추가

            # root (핵심 동사) - deprel: root
            elif word.deprel in ['root', 'compound', 'advcl'] and word.upos == 'VERB':
                # result.append(word.lemma.split('+')[0])  # 원형 추출 후 추가
                result.append(word.text)

            # 꾸미는 말 (주어나 목적어를 꾸미는 말도 포함) - 예: "고민정 최고위원은"
            # 주어나 목적어를 꾸미는 명사
            elif word.deprel in ['compound', 'nmod', 'conj'] and word.upos in ['NOUN', 'CCONJ']:
                result.append(word.lemma.split('+')[0] if len(word.lemma.split(
                    '+')) == 1 else ''.join(word.lemma.split('+')[:-1]))  # 원형 추출 후 추가

    # 리스트를 하나의 문장으로 합치기
    final_result = ' '.join(result) if result else "정보 없음"

    return final_result


def extract_topic(title, body, name, max_length=100):
    """
    기사에서 body만 요약하고 title과 body의 정보를 합쳐서 다시 요약하는 함수
    :param title: 기사의 제목
    :param body: 기사의 본문
    :param name: 주어진 이름 (풀네임 교체에 사용)
    :param max_length: 최종 요약문 최대 길이
    :return: 최종 요약된 문장
    """
    # Step 1: Body만 요약
    body_summary = get_summary(body, max_length=max_length)
    body_summary = replace_name(name, body_summary)  # 이름 교체
    # body_summary = remove_repeated_nouns(body_summary)  # 중복 제거
    body_summary = remove_duplicatates(body_summary)
    # body_summary = ' '.join(extract_keyword(body_summary))

    # Step 2: 제목과 body 요약 합치기
    # combined_text = title + " " + body_summary

    # Step 3: 제목과 body 요약을 합친 내용에 대해 최종 요약 진행
    final_summary = merge_sentences_with_kobart(title, body_summary)

    # 후처리: 이름 교체, 중복 제거, 목적 단순화
    final_summary = replace_name(name, final_summary)
    # final_summary = remove_repeated_nouns(final_summary)
    final_summary = remove_duplicatates(final_summary)

    # final_keyword = " ".join(extract_keyword(final_summary))
    final_keyword = extract_5w1h(final_summary)
    final_keyword = final_summary

    # print("final_summary: ", final_summary)
    # print("final_keyword: ", final_keyword)

    return final_keyword



if __name__ == "__main__":
    # 예시 실행
    title = ""
    body = """
윤석열 전 총장은 이날 국회에서 기자회견을 열고 박주선·김동철 전 의원 영입을 발표하고 "국민 통합을 상징하는 분들을 모시려 노력한 결과 호남을 대표하는 큰 정치인을 캠프에 모시기로 했다"고 밝혔다. 박·김 전 의원은 과거 현 민주당 후보로 광주(光州)에서 국회의원을 했고 각각 국회부의장과 바른미래당 원내대표를 지냈다. 정치권에선 윤 전 총장이 최근 ‘전두환 발언’ 논란으로 악화한 호남 민심 끌어안기에 나선 것 같다는 말이 나왔다. 박·김 전 의원은 "호남에서도 윤 전 총장 리더십을 인정하고 놀라울 정도의 지지를 보낼 것"이라고 했다.
"""
    name = "김동철"

    # 요약 실행
    final_summary = extract_topic(title, body, name)

    # 결과 출력
    print("최종 요약:", final_summary)
