import re
from typing import List

from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    if not text.strip():
        return ""

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False



def chunk_text(
    text: str,
    tokenizer = tokenizer,
    max_tokens: int = 300,
    overlap_sentences: int = 2
) -> List[str]:
    """
    Split text into chunks using sentences and token limits
    """

    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))

        if current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))

            current_chunk = current_chunk[-overlap_sentences:]
            current_tokens = sum(
                len(tokenizer.encode(s, add_special_tokens=False))
                for s in current_chunk
            )

        current_chunk.append(sent)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



def preprocess_text_to_chunks(
    text: str,
    tokenizer = tokenizer,
    max_tokens: int = 300,
    overlap_sentences: int = 2,
    english_only: bool = True
) -> List[str]:
    """
    INPUT: raw text
    OUTPUT: list of cleaned, chunked texts
    """

    text = clean_text(text)

    if english_only and not is_english(text):
        return []

    chunks = chunk_text(
        text=text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_sentences=overlap_sentences
    )

    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

    return chunks


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# raw_text = """
# Apple announced new AI features during its latest event.
# The company focused heavily on on-device intelligence...
# """

# chunks = preprocess_text_to_chunks(
#     text=raw_text,
#     tokenizer=tokenizer,
#     max_tokens=256,
#     overlap_sentences=2
# )

# for i, c in enumerate(chunks):
#     print(f"\n--- Chunk {i} ---\n{c}")