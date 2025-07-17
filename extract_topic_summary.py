# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
# # import nltk
# # import torch
# # import torch.nn.functional as F

# # # NLTK 다운로드
# # nltk.download('punkt')

# # # KoBART 모델과 토크나이저 로드
# # model_t5_dir = "lcw99/t5-base-korean-text-summary"
# # model_KoT5_dir = "noahkim/KoT5_news_summarization"
# # tokenizer_t5 = AutoTokenizer.from_pretrained(model_t5_dir)
# # tokenizer_KoT5 = AutoTokenizer.from_pretrained(model_KoT5_dir)
# # model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_t5_dir)
# # model_KoT5 = AutoModelForSeq2SeqLM.from_pretrained(model_KoT5_dir)

# # max_input_length = 2048


# # class KoBERTSimilarity:
# #     def __init__(self, model_name="skt/kobert-base-v1"):
# #         self.device = torch.device(
# #             "cuda" if torch.cuda.is_available() else "cpu")
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
# #         self.model = AutoModel.from_pretrained(model_name).to(self.device)
# #         self.model.eval()

# #     def get_embedding(self, text):
# #         inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
# #                                 padding=True, max_length=64).to(self.device)
# #         if "token_type_ids" in inputs:
# #             del inputs["token_type_ids"]  # kobert 에러 방지
# #         with torch.no_grad():
# #             outputs = self.model(**inputs)
# #             # [CLS] 토큰의 임베딩
# #             return outputs.last_hidden_state[:, 0, :].squeeze(0)

# #     def similarity(self, text1, text2):
# #         emb1 = self.get_embedding(text1)
# #         emb2 = self.get_embedding(text2)
# #         return F.cosine_similarity(emb1, emb2, dim=0).item()

# #     def most_similar(self, title, candidates):
# #         best_score = -1
# #         best_text = ""
# #         for c in candidates:
# #             score = self.similarity(title, c)
# #             if score > best_score:
# #                 best_score = score
# #                 best_text = c
# #         return best_text, best_score


# # def get_summary_t5(text, max_length=100):
# #     """
# #     주어진 텍스트에 대해 요약을 생성하는 함수
# #     :param text: 요약할 텍스트
# #     :param max_length: 요약문 최대 길이
# #     :return: 요약된 텍스트
# #     """
# #     inputs = [text]
# #     inputs = tokenizer_t5(inputs, max_length=max_input_length,
# #                           truncation=True, return_tensors="pt", padding=True)
# #     output = model_t5.generate(
# #         **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
# #     decoded_output = tokenizer_t5.batch_decode(
# #         output, skip_special_tokens=True)[0]
# #     topic = nltk.sent_tokenize(decoded_output.strip())[0]
# #     return topic


# # def get_summary_KoT5(text, max_length=100):
# #     """
# #     주어진 텍스트에 대해 요약을 생성하는 함수
# #     :param text: 요약할 텍스트
# #     :param max_length: 요약문 최대 길이
# #     :return: 요약된 텍스트
# #     """
# #     inputs = [text]
# #     inputs = tokenizer_KoT5(inputs, max_length=max_input_length,
# #                             truncation=True, return_tensors="pt", padding=True)
# #     output = model_KoT5.generate(
# #         **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
# #     decoded_output = tokenizer_KoT5.batch_decode(
# #         output, skip_special_tokens=True)[0]
# #     topic = nltk.sent_tokenize(decoded_output.strip())[0]
# #     return topic


# # def devide_paragraph(body):
# #     paragraphs = body.split("\n")
# #     return paragraphs


# # def extract_topic(text):
# #     paragraphs = devide_paragraph(text)
# #     # summary = get_summary_KoT5(text, max_length=128)
# #     summary = get_summary_t5(text, max_length=128)
# #     print(summary)
# #     return summary


# # if __name__ == "__main__":
# #     body = """
# #     고민정 최고위원은 CBS 라디오에서 체포동의안 부결 주장에 대해 언급했다.
# #     """

# #     # 요약 실행
# #     final_summary = get_summary(body)

# #     # 결과 출력
# #     print("최종 요약:", final_summary)

# # # ~며, ~면으로 분리
# # # ~했고 분리
# # # 때문에


# # import re
# # import torch
# # import torch.nn.functional as F
# # import stanza
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
# # import nltk

# # # NLTK & Stanza 다운로드
# # nltk.download("punkt")
# # stanza.download("ko")

# # # Stanza 초기화
# # nlp = stanza.Pipeline(lang='ko', processors='tokenize,pos,lemma,depparse')


# # class SentenceSplitter:
# #     CONNECTIVE_PATTERN = r"(면서|며|하며|라며|하고|지만|는데|때문에|이지만|면서도|면서는|고서)"

# #     @staticmethod
# #     def count_verbs(text):
# #         doc = nlp(text)
# #         return sum(1 for sentence in doc.sentences for word in sentence.words if word.upos == "VERB")

# #     @classmethod
# #     def split_by_connectives(cls, text):
# #         parts = re.split(cls.CONNECTIVE_PATTERN, text)
# #         results = []
# #         i = 0
# #         while i < len(parts):
# #             if re.match(cls.CONNECTIVE_PATTERN, parts[i]):
# #                 if i + 1 < len(parts):
# #                     results.append(parts[i + 1].strip())
# #                     i += 2
# #                 else:
# #                     i += 1
# #             else:
# #                 if parts[i].strip():
# #                     results.append(parts[i].strip())
# #                 i += 1
# #         return results

# #     @classmethod
# #     def split_if_needed(cls, text):
# #         return [text.strip()] if cls.count_verbs(text) <= 1 else cls.split_by_connectives(text)


# # class KoBERTSimilarity:
# #     def __init__(self, model_name="skt/kobert-base-v1"):
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
# #         self.model = AutoModel.from_pretrained(model_name).to(self.device)
# #         self.model.eval()

# #     def get_embedding(self, text):
# #         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(self.device)
# #         if "token_type_ids" in inputs:
# #             del inputs["token_type_ids"]
# #         with torch.no_grad():
# #             outputs = self.model(**inputs)
# #         return outputs.last_hidden_state[:, 0, :].squeeze(0)

# #     def similarity(self, text1, text2):
# #         emb1 = self.get_embedding(text1)
# #         emb2 = self.get_embedding(text2)
# #         return F.cosine_similarity(emb1, emb2, dim=0).item()

# #     def most_similar(self, target, candidates):
# #         best_score = -1
# #         best_text = ""
# #         for c in candidates:
# #             score = self.similarity(target, c)
# #             if score > best_score:
# #                 best_score = score
# #                 best_text = c
# #         return best_text, best_score


# # class Summarizer:
# #     def __init__(self, model_dir="lcw99/t5-base-korean-text-summary"):
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
# #         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
# #         self.max_input_length = 2048

# #     def summarize(self, text, max_length=128):
# #         inputs = self.tokenizer([text], max_length=self.max_input_length, truncation=True, return_tensors="pt", padding=True)
# #         output = self.model.generate(**inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
# #         decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
# #         return nltk.sent_tokenize(decoded.strip())[0]


# # class TopicExtractor:
# #     def __init__(self):
# #         self.summarizer = Summarizer()
# #         self.similarity = KoBERTSimilarity()

# #     def extract_topic(self, title, body):
# #         # 1. 요약
# #         summary = self.summarizer.summarize(body)
# #         print(f"\n📌 요약 결과: {summary}")

# #         # 2. 분리
# #         candidates = SentenceSplitter.split_if_needed(summary)
# #         print("\n📌 분리된 문장 후보:")
# #         for i, c in enumerate(candidates, 1):
# #             print(f"  [{i}] {c}")

# #         # 3. 유사도
# #         best_sentence, score = self.similarity.most_similar(title, candidates)
# #         print(f"\n✅ 최종 채택 문장: {best_sentence} (유사도: {score:.4f})")
# #         return best_sentence


# # # 🔍 예시 실행
# # if __name__ == "__main__":
# #     title = "김 의원, 장애인예술단 설립 질의"
# #     body = """
# #     김 의원은 세종시교육청 어울림장애인예술단을 소개했고, 김 교육감에게 장애인예술단 설립에 대해 질의했다.
# #     """

# #     extractor = TopicExtractor()
# #     topic = extractor.extract_topic(title, body)

# import re
# import torch
# import torch.nn.functional as F
# import stanza
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
# import nltk

# # NLTK & Stanza 다운로드
# nltk.download("punkt")
# stanza.download("ko")

# # Stanza 초기화
# nlp = stanza.Pipeline(lang='ko', processors='tokenize,pos,lemma,depparse')


# class SentenceSplitter:
#     CONNECTIVE_PATTERN = r"(하면서도|그러면서|그러나|그런데|하지만|고서|고자|려고|라고도|라며|했고|하며|했다며|면서|이지만|인데)"
#     CONJUNCTION_LEMMAS = {"그러나", "하지만", "게다가", "또한", "그리고", "그러면서"}
#     CCONJ_RELATIONS = {"conj", "cc", "advcl"}

#     @staticmethod
#     def count_verbs(text):
#         doc = nlp(text)
#         return sum(1 for sentence in doc.sentences for word in sentence.words if word.upos == "VERB")

#     @classmethod
#     def split_by_stanza(cls, text):
#         doc = nlp(text)
#         results = []

#         for sent in doc.sentences:
#             split_points = []

#             # 1. 연결어 기반 split
#             for match in re.finditer(cls.CONNECTIVE_PATTERN, sent.text):
#                 split_points.append(match.end())

#             # 2. 접속부사 기반 split
#             for word in sent.words:
#                 if word.upos == "ADV" and word.lemma in cls.CONJUNCTION_LEMMAS:
#                     split_points.append(word.start_char)

#             # 3. 의존구문 기반 split
#             for word in sent.words:
#                 if word.deprel in cls.CCONJ_RELATIONS:
#                     split_points.append(word.start_char)

#             # 정리
#             split_points = sorted(set(split_points))
#             if not split_points:
#                 results.append(sent.text.strip())
#             else:
#                 prev = 0
#                 for idx in split_points:
#                     chunk = sent.text[prev:idx].strip()
#                     if chunk:
#                         results.append(chunk)
#                     prev = idx
#                 last_chunk = sent.text[prev:].strip()
#                 if last_chunk:
#                     results.append(last_chunk)

#         return results

#     @classmethod
#     def split_if_needed(cls, text):
#         verb_count = cls.count_verbs(text)
#         return [text.strip()] if verb_count <= 1 else cls.split_by_stanza(text)


# class KoBERTSimilarity:
#     def __init__(self, model_name="skt/kobert-base-v1"):
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()

#     def get_embedding(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
#                                 padding=True, max_length=64).to(self.device)
#         if "token_type_ids" in inputs:
#             del inputs["token_type_ids"]
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].squeeze(0)

#     def similarity(self, text1, text2):
#         emb1 = self.get_embedding(text1)
#         emb2 = self.get_embedding(text2)
#         return F.cosine_similarity(emb1, emb2, dim=0).item()

#     def most_similar(self, target, candidates):
#         best_score = -1
#         best_text = ""
#         for c in candidates:
#             score = self.similarity(target, c)
#             if score > best_score:
#                 best_score = score
#                 best_text = c
#         return best_text, best_score


# class Summarizer:
#     def __init__(self, model_dir="lcw99/t5-base-korean-text-summary"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
#         self.max_input_length = 2048

#     def summarize(self, text, max_length=128):
#         inputs = self.tokenizer([text], max_length=self.max_input_length,
#                                 truncation=True, return_tensors="pt", padding=True)
#         output = self.model.generate(
#             **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
#         decoded = self.tokenizer.batch_decode(
#             output, skip_special_tokens=True)[0]
#         return nltk.sent_tokenize(decoded.strip())[0]


# class TopicExtractor:
#     def __init__(self):
#         self.summarizer = Summarizer()
#         self.similarity = KoBERTSimilarity()

#     def extract_topic(self, title, body):
#         # 1. 요약
#         summary = self.summarizer.summarize(body)
#         print(f"\n📌 요약 결과: {summary}")

#         # 2. 분리
#         candidates = SentenceSplitter.split_if_needed(summary)
#         print("\n📌 분리된 문장 후보:")
#         for i, c in enumerate(candidates, 1):
#             print(f"  [{i}] {c}")

#         # 3. 유사도 기반 최종 선택
#         best_sentence, score = self.similarity.most_similar(title, candidates)
#         print(f"\n✅ 최종 채택 문장: {best_sentence} (유사도: {score:.4f})")
#         return best_sentence


# # 🔍 예시 실행
# if __name__ == "__main__":
#     title = "김 의원, 장애인예술단 설립 질의"
#     body = """
#     김 의원은 세종시교육청 어울림장애인예술단을 소개했고, 김 교육감에게 장애인예술단 설립에 대해 질의했다.
#     """

#     extractor = TopicExtractor()
#     topic = extractor.extract_topic(title, body)

# import re
# import torch
# import torch.nn.functional as F
# import stanza
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
# import nltk

# # NLTK & Stanza 다운로드
# nltk.download("punkt")
# stanza.download("ko")

# # Stanza 초기화
# nlp = stanza.Pipeline(lang='ko', processors='tokenize,pos,lemma,depparse')


# class SentenceSplitter:
#     CONNECTIVE_PATTERN = r"(면서|며|하며|라며|하고|지만|는데|때문에|이지만|면서도|면서는|고서)"
#     DEP_LABELS = {"conj", "advcl", "parataxis"}

#     @staticmethod
#     def count_verbs(text):
#         doc = nlp(text)
#         return sum(1 for sentence in doc.sentences for word in sentence.words if word.upos == "VERB")

#     @classmethod
#     def split_by_connectives(cls, text):
#         parts = re.split(cls.CONNECTIVE_PATTERN, text)
#         results = []
#         i = 0
#         while i < len(parts):
#             if re.match(cls.CONNECTIVE_PATTERN, parts[i]):
#                 if i + 1 < len(parts):
#                     results.append(parts[i + 1].strip())
#                     i += 2
#                 else:
#                     i += 1
#             else:
#                 if parts[i].strip():
#                     results.append(parts[i].strip())
#                 i += 1
#         return results

#     @classmethod
#     def split_by_dependency(cls, text):
#         doc = nlp(text)
#         for sent in doc.sentences:
#             words = sent.words
#             spans = []

#             for word in words:
#                 if word.deprel in cls.DEP_LABELS:
#                     spans.append(word.start_char)

#             if not spans:
#                 return [text.strip()]

#             spans = sorted(set(spans))
#             spans.append(len(text))
#             prev = 0
#             return [text[prev:sp].strip() for sp in spans if text[prev:sp].strip() and not (prev := sp)]

#         return [text.strip()]

#     @classmethod
#     def split_if_needed(cls, text):
#         verb_count = cls.count_verbs(text)
#         if verb_count <= 1:
#             return [text.strip()]
#         split_by_dep = cls.split_by_dependency(text)
#         if len(split_by_dep) > 1:
#             return split_by_dep
#         return cls.split_by_connectives(text)


# class KoBERTSimilarity:
#     def __init__(self, model_name="skt/kobert-base-v1"):
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()

#     def get_embedding(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
#                                 padding=True, max_length=64).to(self.device)
#         if "token_type_ids" in inputs:
#             del inputs["token_type_ids"]
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].squeeze(0)

#     def similarity(self, text1, text2):
#         emb1 = self.get_embedding(text1)
#         emb2 = self.get_embedding(text2)
#         return F.cosine_similarity(emb1, emb2, dim=0).item()

#     def most_similar(self, target, candidates):
#         best_score = -1
#         best_text = ""
#         if len(candidates) == 1:
#             return candidates[0], 1
#         for c in candidates:
#             score = self.similarity(target, c)
#             if score > best_score:
#                 best_score = score
#                 best_text = c
#         return best_text, best_score


# class Summarizer:
#     def __init__(self, model_dir="lcw99/t5-base-korean-text-summary"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
#         self.max_input_length = 2048

#     def summarize(self, text, max_length=128):
#         inputs = self.tokenizer([text], max_length=self.max_input_length,
#                                 truncation=True, return_tensors="pt", padding=True)
#         output = self.model.generate(
#             **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
#         decoded = self.tokenizer.batch_decode(
#             output, skip_special_tokens=True)[0]
#         return nltk.sent_tokenize(decoded.strip())[0]


# class TopicExtractor:
#     def __init__(self):
#         self.summarizer = Summarizer()
#         self.similarity = KoBERTSimilarity()

#     def extract_topic(self, title, body):
#         summary = self.summarizer.summarize(body)
#         print(f"\n📌 요약 결과: {summary}")

#         candidates = SentenceSplitter.split_if_needed(summary)
#         print("\n📌 분리된 문장 후보:")
#         for i, c in enumerate(candidates, 1):
#             print(f"  [{i}] {c}")

#         best_sentence, score = self.similarity.most_similar(title, candidates)
#         print(f"\n✅ 최종 채택 문장: {best_sentence} (유사도: {score:.4f})")
#         return best_sentence


# # 🔍 예시 실행
# if __name__ == "__main__":
#     title = "김 의원, 장애인예술단 설립 질의"
#     body = """
#     김 의원은 세종시교육청 어울림장애인예술단을 소개했고, 김 교육감에게 장애인예술단 설립에 대해 질의했다.
#     """
#     extractor = TopicExtractor()
#     topic = extractor.extract_topic(title, body)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import stanza

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


class SentenceCleaner:
    def __init__(self):
        self.valid_deprels = {"root", "nsubj", "obj", "iobj", "obl", "ccomp", "conj"}
        self.valid_upos = {"NOUN", "PROPN", "VERB", "PRON"}
        self.valid_xpos = {"ncn", "nq", "ncpa", "xsv", "ep", "ef", "jp", "px", "etm"}

    def is_valid_word(self, word):
        # 형태소 분할
        xpos_parts = word.xpos.split("+")
        
        # 조건 체크
        has_valid_deprel = word.deprel in self.valid_deprels
        has_valid_upos = word.upos in self.valid_upos
        has_valid_xpos = any(x in self.valid_xpos for x in xpos_parts)

        # 교차 조건 2개 이상 만족 시 포함
        return sum([has_valid_deprel, has_valid_upos, has_valid_xpos]) >= 2

    def clean_summary(self, summary: str) -> str:
        doc = nlp(summary)

        for sentence in doc.sentences:
            important_words = []
            for word in sentence.words:
                print(f"{word.text}\t{word.deprel}\t{word.upos}\t{word.xpos}")
                
                if self.is_valid_word(word):
                    important_words.append(word.text)

            if important_words:
                return " ".join(important_words)

        return summary  # fallback


class TopicExtractor:
    def __init__(self):
        self.summarizer = Summarizer()
        self.cleaner = SentenceCleaner()

    def extract_topic(self, title, body):
        summary = self.summarizer.summarize(body)
        print(f"\n📌 요약 결과: {summary}")

        cleaned = self.cleaner.clean_summary(summary)
        print(f"✅ 수식어 제거 후 핵심 문장: {cleaned}")
        return cleaned


# 🔍 예시 실행
if __name__ == "__main__":
    title = "김 의원, 장애인예술단 설립 질의"
    body1 = """
    이번에 의결된 공사법 일부 개정안은 농어촌공사가 해외에서 참여할 수 있는 사업의 종류와 범위를 확대하는 것이 주요 골자다. 공사는 그동안 법적인 제약으로 ‘해외농업개발 및 기술용역사업’에만 참여할 수 있었다. 하지만 이번에 법이 개정되면서 농산업단지와 지역개발, 농어촌용수 및 지하수자원 개발 등 보다 광범위한 분야의 해외사업 참여가 가능해졌다.
공사는 이번 법률개정에 따라 그동안 해외사업을 통해 축적한 경험과 인적 네트워크를 바탕으로 민간기업 등과 연계해 개도국 농촌개발에 적극 나선다는 계획이다.이번 개정안을 대표 발의한 김현권 더불어민주당 의원은 "개발도상국을 중심으로 많은 국가가 국내의 앞선 농산업 기술과 노하우를 전수받길 원하고 있고 국내 민간기업들도 해외진출에 법적인 장벽이 있어 어려움이 많았는데 이를 해소해 주는 것이 국회의 역할"이라며 "이번 법 개정으로 한국농어촌공사가 민간기업과 함께 해외에 진출할 수 있는 길이 열렸기 때문에 앞으로 국내 농산업 시장의 새로운 활로를 모색하고 일자치 창출에 기여할 수 있을 것으로 기대한다"고 말했다.
    """
    body2 = """
    김현권 국회의원(더불어민주당·비례대표·사진)은 "최근 국방부의 통합신공항 부지선정 발표를 환영한다"면서 "앞으로 구미시를 신공항 배후 교통·물류·산업의 중심지로 커 나가도록 지원을 아끼지 않겠다"고 30일 밝혔다.
    """
    body3 = """
    주변 도시를 잇는 교통망 확충 역시 신공항의 성패를 좌우할 핵심과제로 떠오르고 있다. 경북도에 따르면 2021년부터 전철 4곳, 고속도로 2곳 등 총 260㎞에 걸쳐 국비 6조원을 투입하는 신공항과 구미·포항·대구 등 인근 도시들을 연결하는 교통망 확충사업이 추진된다.
김 의원은 "구미시가 신공항배후단지로서 산업·교통·물류의 중심지로 부상하면 구미산단이나 아파트 신도시 활성화뿐만 아니라 도시와 농촌이 조화하는 지역 균형발전이 이뤄질 것"이라고 내다봤다.
    """
    extractor = TopicExtractor()
    topic = extractor.extract_topic(title, body1)
    topic = extractor.extract_topic(title, body2)
    topic = extractor.extract_topic(title, body3)
