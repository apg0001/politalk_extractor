from transformers import BartForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
import stanza
import re
from collections import defaultdict
from text_manager import nlp


class Summarizer:
    def __init__(self, model_dir="lcw99/t5-base-korean-text-summary"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.max_input_length = 2048

    def summarize(self, text, max_length=128):
        inputs = self.tokenizer([text], max_length=self.max_input_length,
                                truncation=True, return_tensors="pt", padding=True)
        output = self.model.generate(
            **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
        decoded = self.tokenizer.batch_decode(
            output, skip_special_tokens=True)[0]
        return nltk.sent_tokenize(decoded.strip())[0]


def restore_names_from_original(original: str, summary: str) -> str:
    def split_words(text):
        return re.findall(r'\b\w+\b', text)

    original_words = split_words(original)
    summary_words = split_words(summary)

    # 2단어씩 묶은 후보들
    original_pairs = [(original_words[i], original_words[i+1])
                      for i in range(len(original_words) - 1)]
    summary_pairs = [(summary_words[i], summary_words[i+1])
                     for i in range(len(summary_words) - 1)]

    # 매핑된 short → full 딕셔너리
    replacement_map = {}

    for o1, o2 in original_pairs:
        for s1, s2 in summary_pairs:
            # short: 김 의원 / full: 김철수 의원
            if o1[0] == s1[0] and o2 == s2:
                short_form = f"{s1} {s2}"
                full_form = f"{o1} {o2}"
                replacement_map[short_form] = full_form

    # 실제 교체 수행
    for short, full in replacement_map.items():
        summary = summary.replace(short, full)

    return summary


class TopicExtractor:
    def __init__(self):
        self.summarizer = Summarizer()

    def extract_topic(self, title, body, purpose, sentence, name):
        summary = self.summarizer.summarize(body)
        print(f"\n요약 결과:\t{summary}")

        replaced = restore_names_from_original(body, summary)

        return replaced


class RedundancyRemover:
    def __init__(self, min_common_len=5):
        self.min_common_len = min_common_len
        self._init_nlp()

    def _init_nlp(self):
        stanza.download('ko')
        # self.nlp = stanza.Pipeline(
        #     lang='ko', processors='tokenize,pos,lemma', verbose=False)
        self.nlp = nlp

    def tokenize(self, text: str):
        doc = self.nlp(text)
        return [word.text for sent in doc.sentences for word in sent.words]

    def lemmatize(self, text: str):
        doc = self.nlp(text)
        return [word.lemma.split('+')[0] for sent in doc.sentences for word in sent.words]

    def trim_redundant_block(self, text: str) -> str:
        tokens = self.tokenize(text)
        lemmas = self.lemmatize(text)

        # lemma -> 모든 등장 인덱스 기록
        lemma_map = defaultdict(list)
        for idx, lemma in enumerate(lemmas):
            lemma_map[lemma].append(idx)

        # 연속된 반복 구간 후보 찾기
        max_start, max_end = -1, -1
        max_len = 0

        for lemma, indices in lemma_map.items():
            if len(indices) < 2:
                continue
            # 모든 가능한 (i, j) 쌍 비교 (i < j)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    start1, start2 = indices[i], indices[j]
                    length = 0
                    while (start1 + length < start2 and
                           start2 + length < len(lemmas) and
                           lemmas[start1 + length] == lemmas[start2 + length]):
                        length += 1
                    if length >= self.min_common_len and length > max_len:
                        max_len = length
                        max_start = start2
                        max_end = start2 + length

        # 제거할 중복 구간이 있다면 제거
        if max_len >= self.min_common_len:
            new_tokens = tokens[:max_start] + tokens[max_end:]
            return ' '.join(new_tokens)
        return text


# 🔍 예시 실행
if __name__ == "__main__":
    title = "김 의원, 장애인예술단 설립 질의"
    body1 = """
    통합당 간사인 이채익 의원은 코로나19 자가격리자에게 거소투표·선상투표를 허용하는 등 대책 마련을 주문했고, 이승택 후보자는 "보건당국의 이동제한 허용을 전제로 사전투표가 가능할 것 같다"면서 "참정권 확대라는 부분과 관련해서 적극 의견을 개진하겠다"고 답했다.
민주당 권미혁 의원은 "전자거소투표 도입을 검토해야 한다"며 선관위의 온라인투표시스템인 '케이보팅'(K-voting) 이용 방안을 제안했다.
    """
#     body2 = """
#     김현권 국회의원(더불어민주당·비례대표·사진)은 "최근 국방부의 통합신공항 부지선정 발표를 환영한다"면서 "앞으로 구미시를 신공항 배후 교통·물류·산업의 중심지로 커 나가도록 지원을 아끼지 않겠다"고 30일 밝혔다.
#     """
#     body3 = """
#     주변 도시를 잇는 교통망 확충 역시 신공항의 성패를 좌우할 핵심과제로 떠오르고 있다. 경북도에 따르면 2021년부터 전철 4곳, 고속도로 2곳 등 총 260㎞에 걸쳐 국비 6조원을 투입하는 신공항과 구미·포항·대구 등 인근 도시들을 연결하는 교통망 확충사업이 추진된다.
# 김 의원은 "구미시가 신공항배후단지로서 산업·교통·물류의 중심지로 부상하면 구미산단이나 아파트 신도시 활성화뿐만 아니라 도시와 농촌이 조화하는 지역 균형발전이 이뤄질 것"이라고 내다봤다.
#     """
    extractor = TopicExtractor()
    topic = extractor.extract_topic(title, body1)
    # topic = extractor.extract_topic(title, body2)
    # topic = extractor.extract_topic(title, body3)
