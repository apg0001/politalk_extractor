from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# NLTK 다운로드
nltk.download('punkt')

# KoBART 모델과 토크나이저 로드
model_dir = "lcw99/t5-base-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 2048


def get_summary(text, max_length=100):
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

def devide_paragraph(body):
    paragraphs = body.split("\n")
    return paragraphs

def extract_topic(text):
    paragraphs = devide_paragraph(text)
    summary = get_summary(text)
    print(summary)
    return summary

if __name__ == "__main__":
    body = """
    고민정 최고위원은 CBS 라디오에서 체포동의안 부결 주장에 대해 언급했다.
    """

    # 요약 실행
    final_summary = get_summary(body)

    # 결과 출력
    print("최종 요약:", final_summary)
