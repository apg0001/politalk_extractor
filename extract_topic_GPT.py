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
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt", padding=True)
    output = model.generate(**inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    topic = nltk.sent_tokenize(decoded_output.strip())[0]
    return topic

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
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True, padding=True)
    
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_article(title, body, name, max_length=30):
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
    body_summary = remove_repeated_nouns(body_summary)  # 중복 제거
    
    # Step 2: 제목과 body 요약 합치기
    combined_text = title + " " + body_summary

    # Step 3: 제목과 body 요약을 합친 내용에 대해 최종 요약 진행
    final_summary = merge_sentences_with_kobart(title, combined_text)

    # 후처리: 이름 교체, 중복 제거, 목적 단순화
    final_summary = replace_name(name, final_summary)
    final_summary = remove_repeated_nouns(final_summary)

    return final_summary


# 예시 실행
title = "고민정 최고위원, 체포동의안 부결 주장"
body = "고민정 최고위원은 CBS 라디오에서 체포동의안 부결 주장에 대해 언급했다."
name = "고민정"

# 요약 실행
final_summary = summarize_article(title, body, name)

# 결과 출력
print("최종 요약:", final_summary)