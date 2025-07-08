from text_manager import nlp, simplify_purpose
import re
from gensim.models import KeyedVectors

import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel, pipeline


class Modifier:
    # w2v_vec_path = "./csv2excel/ko_reformatted.vec"
    w2v_vec_path = "./code/csv2excel/ko_reformatted.vec"
    w2v_model = KeyedVectors.load_word2vec_format(w2v_vec_path, binary=False)

    kobert_model_name = "skt/kobert-base-v1"
    kobert_tokenizer = AutoTokenizer.from_pretrained(kobert_model_name)
    kobert_model = AutoModel.from_pretrained(kobert_model_name).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    kobert_model.eval()

    SYNONYM_DICT = {
        "통과": "처리",
        "기소": "재판에 넘겨져",
        "무산": "불발",
        "출마": "입후보",
        "성토장": "비판 현장",
        "불꽃튀는": "치열한",
        "지지선언": "지지 입장 표명",
        "당선": "선출",
        "노조": "노동조합",
        "전 의원": "前 의원",
        "전 한화 수석코치": "前 한화 수석코치",
        "설립안": "설립 계획",
        "복합화": "통합 개발",
        "의도적 창작": "고의적 조작",
        "문화예술": "예술 활동",
        "만끽": "즐김",
        "동성": "同性",
        "사업중단 위기": "운영 차질",
        "불꽃튀는 맞대결": "치열한 경쟁",
        "한목소리 질타": "공동 비판",
        "톤다운": "수위 조절",
        "타이밍": "시점",
        "연정을": "연합 정부를",
        "재도약": "부활",
        "탈당": "당을 떠나",
        "지위 잃나": "자격 상실 우려",
        "실형": "형 확정",
        "때리는": "비판하는",
        "실명 저격": "이름을 거론하며 비난",
        "영입": "포섭",
        "진출 박차": "적극 추진",
        "독오른": "분노한",
        "열공중": "열심히 공부 중",
        "적색경보": "위기 경고",
        "폐쇄": "종료",
        "환급": "돌려받음",
        "배출": "탄생",
        "술렁": "반향",
        "졸속": "부실",
        "단일화 파기": "단일화 실패",
        "추경": "추가경정예산",
        "불리기 싫다": "호칭을 거부했다",
        "눈물 사연": "감정적인 이야기",
        "재산공개": "자산 공개",
        "평균재산": "평균 자산",
        "외국인": "해외 인력",
        "부활": "재기",
        "지원": "지원을 약속",
        "허문다": "무너뜨린다",
        "예비": "대비",
        "대거": "속속",
        "맞대결": "경합",
    }
    
    # 바뀌면 안되는 단어들
    not_replace = ["검찰", "장", "당", "구", "업", "회", "사", "원", "관", "자", "법", "국민의힘", "에", "메카", "신인"]

    @staticmethod
    def replace_quotes(text: str):
        # 정규 따옴표로 변경
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("‘", "'")
        text = text.replace("’", "'")
        return text

    @staticmethod
    def normalize_text(text: str):
        # 꺽쇠 괄호 안 텍스트 제거
        pattern = r'\[.*?\]|<.*?>'
        text = re.sub(pattern, '', text)
        # ... 제거
        text = text.replace("...", " ")
        text = text.replace("…", " ")
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text

    def get_replaceable_words(text):
        doc = nlp(text)
        candidates = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in ['ADJ', 'ADV', 'ADP', 'VERB', 'NOUN']:
                    # if word.
                    lemma = word.lemma.split('+')[0]  # 복합 형태소에서 첫 성분
                    candidates.append((word.text, lemma, word.upos))
        return candidates

    @classmethod
    def add_verb(cls, text):
        # 예: “XXX” → “XXX”라고 언급
        return text + "라고 언급"

    @classmethod
    def change_word_order(cls, text):
        quote_count = text.count('"')

        if quote_count == 0:
            return text  # 큰따옴표 없음 → 그대로 반환

        elif quote_count == 2:
            first_quote_idx = text.find('"')
            second_quote_idx = text.find('"', first_quote_idx + 1)

            quote_content = text[first_quote_idx:second_quote_idx + 1].strip()
            before = text[:first_quote_idx].strip()
            after = text[second_quote_idx + 1:].strip()

            # Case 1: "A" B → B "A"
            if first_quote_idx == 0:
                return f"{after} {quote_content}"

            # Case 2: B "A" → "A" B
            elif second_quote_idx == len(text) - 1:
                return f"{quote_content} {before}"

            # 중간 위치는 원문 그대로 반환
            else:
                # 모든 큰따옴표 문장 추출
                quote_contents = re.findall(r'"[^"]+"', text)
                # 본문에서 큰따옴표 문장 제거
                text_wo_quotes = text
                for q in quote_contents:
                    text_wo_quotes = text_wo_quotes.replace(q, '').strip()
                text_wo_quotes = re.sub(r'\s+', ' ', text_wo_quotes).strip()

                # 큰따옴표 문장 뒤로 이동
                return f"{text_wo_quotes} {' '.join(quote_contents)}"

        elif quote_count > 2:
            # 모든 큰따옴표 문장 추출
            quote_contents = re.findall(r'"[^"]+"', text)
            # 본문에서 큰따옴표 문장 제거
            text_wo_quotes = text
            for q in quote_contents:
                text_wo_quotes = text_wo_quotes.replace(q, '').strip()
            text_wo_quotes = re.sub(r'\s+', ' ', text_wo_quotes).strip()

            # 큰따옴표 문장 뒤로 이동
            return f"{text_wo_quotes} {' '.join(quote_contents)}"

    @classmethod
    def replace_suffix(cls, text: str):
        text = text.replace("한다", "하기로")
        return text

    @classmethod
    def replace_with_dictionary(cls, text: str) -> str:
        # 긴 key부터 우선 매칭
        for key, value in sorted(cls.SYNONYM_DICT.items(), key=lambda x: -len(x[0])):
            if key in text:
                text = text.replace(key, value)
        return text

    # @classmethod
    # def replace_with_synonyms(cls, text, model, topn=5):
    #     replaceables = cls.get_replaceable_words(text)
    #     best_score = -1
    #     best_candidate = None

    #     for original, lemma, pos in replaceables:
    #         try:
    #             # 유사도 상위 N개를 가져와 비교
    #             similar_words = model.most_similar(lemma, topn=topn)
    #             if not similar_words:
    #                 continue

    #             # 가장 유사한 단어 선택
    #             word, score = similar_words[0]
    #             if score > best_score:
    #                 best_score = score
    #                 best_candidate = (original, word)
    #         except KeyError:
    #             continue  # 사전에 없는 단어는 스킵

    #     # 유사도가 가장 높은 단어 하나만 치환
    #     if best_candidate:
    #         original, replacement = best_candidate
    #         replaced = re.sub(r'\b' + re.escape(original) +
    #                           r'\b', replacement, text)
    #         return replaced

    #     return text  # 대체할 게 없으면 원문 그대로 반환

    # @classmethod
    # def replace_with_synonyms(cls, text, model, topn=5):
    #     replaceables = cls.get_replaceable_words(text)
    #     replaced = text

    #     for original, lemma, pos in replaceables:
    #         try:
    #             similar_word = model.most_similar(lemma, topn=topn)[0][0]
    #             if (lemma[-1] == similar_word[-1]):
    #                 # 끝 글자가 같은 경우는 교체하지 않음
    #                 # ex) 남자<->여자, 포유류<->양서류 등
    #                 continue
    #             if (lemma.endswith(("검찰", "장", "당", "구", "업", "회", "사", "원", "관", "자"))):
    #                 # 바뀌면 안되는 단어들
    #                 continue

    #             # 원래 단어가 텍스트에 있을 경우 대체
    #             replaced = re.sub(r'\b' + re.escape(original) +
    #                               r'\b', similar_word, replaced)
    #         except KeyError:
    #             continue  # Word2Vec 사전에 없는 단어는 스킵

    #     return replaced
    @classmethod
    def get_embedding_kobert(cls, text):
        inputs = cls.kobert_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        with torch.no_grad():
            outputs = cls.kobert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze()

    # @classmethod
    # def replace_with_synonyms(cls, text, model, topn=5):
    #     replaceables = cls.get_replaceable_words(text)
    #     base_emb = cls.get_embedding_kobert(text)
    #     best_candidate = None
    #     best_score = -1

    #     for original, lemma, pos in replaceables:
    #         try:
    #             similar_words = model.most_similar(lemma, topn=topn)
    #             # print(model.most_similar(lemma, topn=topn))
    #             for word, _ in similar_words:
    #                 # print(word)
    #                 if (lemma[-1] == word[-1]):
    #                     # 끝 글자가 같은 경우는 교체하지 않음
    #                     # ex) 남자<->여자, 포유류<->양서류 등
    #                     continue
    #                 if (lemma.endswith(("검찰", "장", "당", "구", "업", "회", "사", "원", "관", "자"))):
    #                     # 바뀌면 안되는 단어들
    #                     continue

    #                 # 원래 단어가 텍스트에 있을 경우 대체
    #                 replaced = re.sub(r'\b' + re.escape(original) +
    #                                 r'\b', word, text)
    #                 cand_emb = cls.get_embedding_kobert(replaced)
    #                 score = F.cosine_similarity(base_emb, cand_emb, dim=0).item()
    #                 print(f"{original}({lemma}) -> {word}: {replaced}, {score}")
    #                 if score > best_score:
    #                     best_score = score
    #                     best_candidate = (original, word)
    #         except KeyError:
    #             continue  # Word2Vec 사전에 없는 단어는 스킵

    #     if best_candidate:
    #         return re.sub(r'\b' + re.escape(best_candidate[0]) + r'\b', best_candidate[1], text)
    #     return text

    @classmethod
    def replace_with_synonyms(cls, text, model, topn=5):
        replaced = text
        replaceables = cls.get_replaceable_words(text)
        base_emb = cls.get_embedding_kobert(text)
        replace_word = dict()

        for original, lemma, pos in replaceables:
            best_score = -1
            try:
                # similar_words = model.most_similar(lemma, topn=topn)
                similar_words = model.most_similar(original, topn=topn)
                print(similar_words)
                for word, _ in similar_words:
                    if (lemma[-1] == word[-1]):
                        # 끝 글자가 같은 경우는 교체하지 않음
                        # ex) 남자<->여자, 포유류<->양서류 등
                        continue
                    # if (lemma.endswith(("검찰", "장", "당", "구", "업", "회", "사", "원", "관", "자"))):
                    if any(sub in original or original in sub for sub in cls.not_replace):
                        continue

                    # 원래 단어가 텍스트에 있을 경우 대체
                    temp = re.sub(r'\b' + re.escape(original) +
                                  r'\b', word, text)
                    cand_emb = cls.get_embedding_kobert(temp)
                    score = F.cosine_similarity(
                        base_emb, cand_emb, dim=0).item()
                    if score > best_score:
                        best_score = score
                        replace_word[original] = word
                        print(f"({original}) -> {word}, {score}")
            except KeyError:
                continue  # Word2Vec 사전에 없는 단어는 스킵
            
        print(replace_word)
        for key, value in replace_word.items():
            replaced = re.sub(r'\b' + re.escape(key) +
                    r'\b', value, replaced)

        return replaced

    @classmethod
    def modify_title(cls, text):
        text = cls.replace_quotes(text)
        text = cls.normalize_text(text)
        topic = text

        # 큰따옴표로 끝나는 경우
        # 마지막에 "~라고 언급" 추가
        if text.endswith("\""):
            # print("case1")
            topic = cls.add_verb(text)

        # 큰따옴표가 포함되어 있고 큰따옴표가 마지막이 아닌 경우
        # + 큰따옴표 문장이 한 개인 경우
        # 어순을 바꾼 뒤 큰따옴표로 끝나면 "~라고 발언" 추가
        elif "\"" in text:
            # print("case2")
            topic = cls.change_word_order(text)
            # 어순 바꾼 뒤 큰따옴표로 끝나면 "~라고 발언" 추가
            if topic.endswith("\""):
                topic = cls.add_verb(topic)

        # 임의 딕셔너리 사용
        else:
            # print("case3")
            topic = cls.replace_with_dictionary(text)

        # 어떤 조건에도 포함되지 않는 경우 동의어·유의어 사전 활용
        if text == topic:
            # print("case4")
            topic = cls.replace_with_synonyms(text, cls.w2v_model)
            topic = topic

        # ~한다 -> ~하기로 어미 변경
        # topic = cls.replace_suffix(topic)
        topic = simplify_purpose(topic, "")

        return cls.normalize_text(topic.strip())


# from text_manager import nlp

# def get_raw_word(text):
#     doc = nlp(text)
#     for sentence in doc.sentences:
#         print(sentence)

class Paraphrase:
    # 장치 설정 (GPU가 있으면 GPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 토크나이저 로드
    model = BartForConditionalGeneration.from_pretrained(
        'guialfaro/korean-paraphrasing').to(device)
    tokenizer = AutoTokenizer.from_pretrained('guialfaro/korean-paraphrasing')

    @classmethod
    def paraphrase_text(cls, sentence, model=model, tokenizer=tokenizer, device=device):
        """
        주어진 문장을 바탕으로 문장을 패러프레이징하여 반환하는 함수
        """
        text = f"paraphrase: {sentence} "

        encoding = tokenizer.batch_encode_plus(
            [text],
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = encoding["input_ids"].to(device, dtype=torch.long)
        source_mask = encoding["attention_mask"].to(device, dtype=torch.long)

        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        preds = [tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        return preds[0]


class Paraphrase2:
    # 모델 이름 설정
    model_name = "psyche/KoT5-paraphrase-generation"

    # pipeline을 사용하여 모델과 토크나이저 불러오기
    generator = pipeline("text2text-generation",
                         model=model_name, device=0)  # device=0은 GPU 사용 시

    @classmethod
    def generate(cls, prompt, max_tokens=128):
        # 텍스트 생성
        # result = cls.generator(prompt, max_length=512, num_return_sequences=1, max_new_tokens=max_tokens)
        result = cls.generator(prompt, max_length=512, num_return_sequences=1)
        # print(result)

        # 생성된 텍스트 반환
        return result[0]['generated_text']

def test(text):
    model_output = Paraphrase2.generate(text)
    normalized_model_output = Paraphrase2.generate(Modifier.normalize_text(text))
    rule_output = Modifier.modify_title(text)
    rule_then_model_output = Paraphrase2.generate(rule_output)
    
    result = (
                f"모델:\t\t{model_output}\n"
                f"정규화+모델\t{normalized_model_output}\n"
                f"사전:\t\t{rule_output}\n"
                f"사전+모델:\t{rule_then_model_output}\n"
                f"-------------------------------------------------------------\n"
            )

    return result

if __name__ == "__main__":
    titles = [
        "친문·친조국 세력의 윤석열 검찰 때리기, 法治 허문다",
        "민주당 경선 1라운드는 ‘5분 압박면접’…정봉주는 ‘부적격’",
        "친명 신인 vs 재선 의원 ‘무연고 맞대결’ [심층기획-4·10 총선 격전지를 가다]",
        "‘이재명 사법리스크 관여’ 대장동 변호사들 대거 국회 입성",
        "김병관 전 민주당 의원, 동성 강제추행 혐의로 기소돼",
        "재산 형성·음주운전 ‘송곳질문’에… 예비 후보들 진땀",
        "494억 최고 자산가 기재부관리관…금융위 부위원장 200억 줄었다[재산공개]",
        "거칠어진 與 경선…끝나도 ‘원팀’ 가능할까",
        "손학규 ‘마이웨이’… 민주평화당·대안신당과 ‘호남 통합당’ 추진",
        "김종훈 울산 동구청장 ‘1호 사업’ 결실 봤다",
        "조선산업 메카 울산 동구 지역특산물 ‘용가자미’도 수출길 오른다",
        "‘노동자의 텃밭’ 울산 동구청장, 국민의힘-진보당 양자대결"
    ]

    for t in titles:
        original = t
        model_output = Paraphrase2.generate(t)
        normalized_model_output = Paraphrase2.generate(Modifier.normalize_text(t))
        rule_output = Modifier.modify_title(t)
        rule_then_model_output = Paraphrase2.generate(rule_output)

        result = (
            f"\n원문:    {original}\n"
            f"모델:      {model_output}\n"
            f"정규화+모델: {normalized_model_output}\n"
            f"사전:      {rule_output}\n"
            f"사전+모델:  {rule_then_model_output}\n"
        )

        print(result)