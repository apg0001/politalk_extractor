from transformers import BartForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
import stanza
import re

# NLTK, Stanza 초기화
nltk.download('punkt')
stanza.download('ko')
nlp = stanza.Pipeline(lang='ko', processors='tokenize,pos,lemma,depparse')


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


# class SentenceCleaner:
#     def __init__(self):
#         self.valid_deprels = {"root", "nsubj",
#                               "obj", "iobj", "obl", "ccomp", "conj", "xcomp", "advcl", "acl", "nmod", "compound", "conj", "dislocated", "dep", "aux"}
#         self.valid_upos = {"NOUN", "PROPN", "VERB", "PRON", "AUX", "CCONJ", "SCONJ", "CONJ", "ADV", "ADJ"}
#         self.valid_xpos = {}  # {"ncn", "nq", "ncpa", "xsv", "ep", "ef", "jp", "px", "etm"}

#     def is_valid_word(self, word):
#         # 형태소 분할
#         xpos_parts = word.xpos.split("+")

#         # 조건 체크
#         has_valid_deprel = word.deprel in self.valid_deprels
#         has_valid_upos = word.upos in self.valid_upos
#         has_valid_xpos = any(x in self.valid_xpos for x in xpos_parts)

#         # 교차 조건 2개 이상 만족 시 포함
#         return sum([has_valid_deprel, has_valid_upos, has_valid_xpos]) >= 2

#     def clean_summary(self, summary: str) -> list:
#         doc = nlp(summary)

#         for sentence in doc.sentences:
#             important_words = []
#             for word in sentence.words:
#                 # print(f"{word.text}\t{word.deprel}\t{word.upos}\t{word.xpos}")

#                 if self.is_valid_word(word):
#                     important_words.append(word.text)
#                     if((word.upos in {"CONJ", "SCONJ", "CCONJ"} and word.deprel in {"conj", "root", "obl", "advcl", "cc"})):
#                     #    or \
#                         # (word.upos == "AUX" and word.deprel == "aux" and word[word.id].upos not in {"CONJ", "SCONJ", "CCONJ", "AUX"} and word.text not in {"있는"})):
#                         break

#             if important_words:
#                 return " ".join(important_words)

#         return summary  # fallback
# class SentenceCleaner:
#     def __init__(self):
#         self.valid_deprels = {
#             "root", "nsubj", "obj", "iobj", "obl", "ccomp", "conj", "xcomp",
#             "advcl", "acl", "nmod", "compound", "dislocated", "dep", "aux"
#         }
#         self.valid_upos = {
#             "NOUN", "PROPN", "VERB", "PRON", "AUX", "CCONJ", "SCONJ", "CONJ", "ADV", "ADJ"
#         }
#         self.valid_xpos = set()  # 비워둠. 조건 2개 이상이면 통과됨.

#         self.connecting_deprels = {"conj", "root", "obl", "advcl", "cc"}
#         self.connecting_upos = {"CONJ", "SCONJ", "CCONJ"}

#     def is_valid_word(self, word):
#         xpos_parts = word.xpos.split("+")
#         has_valid_deprel = word.deprel in self.valid_deprels
#         has_valid_upos = word.upos in self.valid_upos
#         has_valid_xpos = any(x in self.valid_xpos for x in xpos_parts)
#         return sum([has_valid_deprel, has_valid_upos, has_valid_xpos]) >= 2

#     def clean_summary(self, summary: str) -> list[str]:
#         doc = nlp(summary)

#         for sentence in doc.sentences:
#             front = []
#             back = []
#             switching = False

#             for word in sentence.words:
#                 if self.is_valid_word(word):
#                     if (word.upos in self.connecting_upos and word.deprel in self.connecting_deprels):
#                         switching = True
#                         continue  # 연결어는 제외

#                     if switching:
#                         back.append(word.text)
#                     else:
#                         front.append(word.text)

#             return [" ".join(front).strip(), " ".join(back).strip()]

#         return [summary.strip(), ""]  # fallback: 나누지 못한 경우

# class SentenceCleaner:
#     def __init__(self):
#         self.valid_deprels = {
#             "root", "nsubj", "obj", "iobj", "obl", "ccomp", "conj", "xcomp",
#             "advcl", "acl", "nmod", "compound", "dislocated", "dep", "aux"
#         }
#         self.valid_upos = {
#             "NOUN", "PROPN", "VERB", "PRON", "AUX", "CCONJ", "SCONJ", "CONJ", "ADV", "ADJ"
#         }
#         self.valid_xpos = set()

#         # clause를 분리할 때 기준이 되는 연결 표현들
#         self.split_keywords = {"그리고", "그러나", "하지만", "또는", "또"}

#     def is_valid_word(self, word):
#         xpos_parts = word.xpos.split("+")
#         has_valid_deprel = word.deprel in self.valid_deprels
#         has_valid_upos = word.upos in self.valid_upos
#         has_valid_xpos = any(x in self.valid_xpos for x in xpos_parts)
#         return sum([has_valid_deprel, has_valid_upos, has_valid_xpos]) >= 2

#     def clean_summary(self, summary: str) -> list[str]:
#         doc = nlp(summary)

#         for sentence in doc.sentences:
#             return self.split_summary_clauses(sentence)

#         return [summary.strip(), ""]

#     def split_summary_clauses(self, sentence) -> list[str]:
#         """
#         root 외에 conj/advcl/ccomp에 해당하는 VERB가 있는 경우, 그 지점을 기준으로 나눈다.
#         또는 '그리고', '그러나' 같은 연결어로도 분리.
#         """
#         front = []
#         back = []
#         switching = False
#         root_ids = set()

#         # 먼저 root 및 병렬 절 후보 탐색
#         for word in sentence.words:
#             if word.deprel == "root":
#                 root_ids.add(word.id)
#             if word.head in root_ids and word.upos == "VERB" and word.deprel in {"conj", "advcl", "ccomp"}:
#                 root_ids.add(word.id)

#         for word in sentence.words:
#             if not self.is_valid_word(word):
#                 continue

#             # 텍스트 기반 분리: 연결어
#             if word.text in self.split_keywords:
#                 switching = True
#                 continue

#             # 구조 기반 분리: root 외 병렬 동작 발견
#             if word.id in root_ids and word.deprel != "root":
#                 if switching == False:
#                     front.append(word.text)
#                 else:
#                     back.append(word.text)
#                 switching = True
#                 continue

#             if switching:
#                 back.append(word.text)
#             else:
#                 front.append(word.text)

#         return [" ".join(front).strip(), " ".join(back).strip()]


# class SummarySelector:
#     def __init__(self):
#         stanza.download("ko", processors="tokenize,pos,lemma", verbose=False)
#         self.nlp = stanza.Pipeline(
#             "ko", processors="tokenize,pos,lemma", verbose=False)

#     def get_lemmas(self, text):
#         """문장에서 lemma(표제어) 집합 추출"""
#         doc = self.nlp(text)
#         lemmas = set()
#         for sentence in doc.sentences:
#             for word in sentence.words:
#                 lemmas.add(word.lemma)
#         return lemmas

#     def jaccard_similarity(self, lemmas1, lemmas2):
#         """두 집합 간 자카드 유사도 계산"""
#         intersection = lemmas1 & lemmas2
#         union = lemmas1 | lemmas2
#         if not union:
#             return 0.0
#         return len(intersection) / len(union)

#     def select_least_similar(self, summary_list, purpose, speech):
#         """가장 덜 유사한 요약문 선택 (평균 유사도 50% 넘으면 빈칸 반환)"""
#         purpose_lemmas = self.get_lemmas(purpose)
#         speech_lemmas = self.get_lemmas(speech)

#         min_similarity = float("inf")
#         selected_summary = None

#         for summary in summary_list:
#             summary_lemmas = self.get_lemmas(summary)
#             sim_purpose = self.jaccard_similarity(
#                 summary_lemmas, purpose_lemmas)
#             sim_speech = self.jaccard_similarity(summary_lemmas, speech_lemmas)
#             avg_similarity = (sim_purpose + sim_speech) / 2

#             if avg_similarity < min_similarity:
#                 min_similarity = avg_similarity
#                 selected_summary = summary

#         # 평균 유사도가 0.5 이상이면 빈칸 반환
#         if min_similarity >= 0.5:
#             return ""
#         if min_similarity == avg_similarity:
#             if summary_list[1] != "":
#                 return summary_list[1]

#         return selected_summary


# class Paraphraser:
#     # 모델 이름 설정
#     model_name = "psyche/KoT5-paraphrase-generation"

#     # pipeline을 사용하여 모델과 토크나이저 불러오기
#     generator = pipeline("text2text-generation",
#                          model=model_name, device=0)  # device=0은 GPU 사용 시

#     @classmethod
#     def generate(cls, prompt, max_tokens=128):
#         # 텍스트 생성
#         # result = cls.generator(prompt, max_length=512, num_return_sequences=1, max_new_tokens=max_tokens)
#         result = cls.generator(prompt, max_length=512, num_return_sequences=1)
#         # print(result)

#         # 생성된 텍스트 반환
#         return result[0]['generated_text']

# def replace_name(name, text):
#     """
#     텍스트에서 주어진 이름을 풀네임으로 교체
#     성씨 뒤에 직책이나 직위가 있을 경우 해당 직위도 함께 교체
#     """
#     full_name = name
#     position_words = [
#         "대통령", "국회의원", "국회위원", "최고위원", "당대표", "대변인", "부대표", "비례대표", "원내대표", "전 대표",
#         "전 의원", "당수", "시의원", "지방의회 의원", "장관", "부장관", "청와대 비서실장", "청와대 대변인", "통일부 장관",
#         "경제부총리", "인사혁신처장", "외교부 장관", "법무부 장관", "교육부 장관", "노동부 장관", "사회복지부 장관",
#         "지방자치단체장", "행정자치부 장관", "선거관리위원회 위원", "정치인", "정부 고위 관계자", "정당 대표", "정당 최고위원",
#         "정당 대변인", "정당 부대표",
#     ]
#     # 의존 구문 분석
#     doc = nlp(text)
    
#     # 텍스트에서 각 단어를 분석하여 직위가 있을 경우 풀네임으로 교체
#     for sentence in doc.sentences:
#         for i, word in enumerate(sentence.words):
#             if word.text == name[0] and i + 1 < len(sentence.words) and any(pos_word in sentence.words[i+1].text for pos_word in position_words):
#                 word.text = full_name

#         # 수정된 텍스트 반환
#         summary = " ".join([word.text for word in sentence.words])
#         summary = re.sub(r'\s+\.', '.', string=summary)
#     return summary


def restore_names_from_original(original: str, summary: str) -> str:
    def split_words(text):
        return re.findall(r'\b\w+\b', text)

    original_words = split_words(original)
    summary_words = split_words(summary)

    # 2단어씩 묶은 후보들
    original_pairs = [(original_words[i], original_words[i+1]) for i in range(len(original_words) - 1)]
    summary_pairs = [(summary_words[i], summary_words[i+1]) for i in range(len(summary_words) - 1)]

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
        # self.cleaner = SentenceCleaner()
        # self.paraphraser = Paraphraser()
        # self.selector = SummarySelector()
        # self.paraphraser2 = Paraphrase()

    def extract_topic(self, title, body, purpose, sentence, name):
        summary = self.summarizer.summarize(body)
        print(f"\n요약 결과:\t{summary}")

        # cleaned = self.cleaner.clean_summary(summary)
        # print(f"수식어 제거:\t{cleaned}")

        # if cleaned[1] != "":
        #     selected = self.selector.select_least_similar(
        #         cleaned, purpose, sentence)
        # else:
        #     selected = cleaned[0]
        
        replaced = restore_names_from_original(body, summary)

        # paraphrased = self.paraphraser.generate(cleaned)
        # print(f"paraphrase:\t{paraphrased}")

        return replaced


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
