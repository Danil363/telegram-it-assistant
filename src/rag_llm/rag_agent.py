from pathlib import Path
import sys
import random
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from ingestion.web_search import collect_info
from preprocessing.preprocess_text import preprocess_text_to_chunks

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from transformers import AutoModelForCausalLM,  AutoTokenizer

import torch

IT_TOPIC = """
Information technology, software development, programming, artificial intelligence,
machine learning, computer science, cloud computing, cybersecurity,
data science, databases, networking, operating systems
"""

OFF_TOPIC_RESPONSES = [
    "I specialize in IT and technology topics. Try asking about AI, programming, or software development.",
    "That question seems outside my expertise. I can help with IT, AI, and tech-related topics.",
    "I'm focused on technology and IT. Ask me something about software, AI, or computers.",
    "Interesting question! However, I only handle IT and technology-related topics.",
    "I don't have enough knowledge outside IT topics, but I'd be happy to help with tech questions."
]

IT_SIMILARITY_THRESHOLD = 0.25
SEARCH_SIMILARITY_THRESHOLD = 0.35

def estimate_max_tokens(query: str, context: str) -> int:
    base = 128

    if len(context) > 2000:
        base = 256
    if len(context) > 4000:
        base = 384

    if any(w in query.lower() for w in ["explain", "difference", "how", "why"]):
        base += 64

    return min(base, 512)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Вычисляет косинусную схожесть между двумя векторами."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_query_about_it(
    query: str, 
    sentence_transformer: SentenceTransformer,
    threshold: float = IT_SIMILARITY_THRESHOLD
) -> Tuple[float, bool]:
    """Определяет, относится ли запрос к IT-тематике."""
    topic_emb = sentence_transformer.encode(IT_TOPIC)
    query_emb = sentence_transformer.encode(query)
    similarity = cosine_similarity(topic_emb, query_emb)
    return similarity, similarity >= threshold

def make_index(texts, sentence_transformer, save_path=None):
    if not texts:
        raise ValueError("No texts to index")

    embeddings = sentence_transformer.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # if save_path:
    #     faiss.write_index(index, save_path)

    return index


def search_in_index(
    query: str,
    index: faiss.Index,
    sentence_transformer: SentenceTransformer,
    texts: List[str] = None,
    df = None,
    k: int = 3,
    similarity_threshold: float = SEARCH_SIMILARITY_THRESHOLD
) -> List[str]:
    """Ищет релевантные чанки в индексе."""
    q_emb = sentence_transformer.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    D, I = index.search(q_emb, k)
    context_chunks = []
    
    for idx, score in zip(I[0], D[0]):
        if score > similarity_threshold:
            if df is not None:
                context_chunks.append(df.iloc[idx]['text'])
            elif texts is not None:
                context_chunks.append(texts[idx])
    
    return context_chunks

def get_off_topic_response() -> str:
    """Возвращает случайный ответ для off-topic запросов."""
    return random.choice(OFF_TOPIC_RESPONSES)

def rag_agent(
    query: str,
    model,
    tokenizer,
    sentence_transformer: SentenceTransformer,
    df = None,
    index_path: str = 'data/tech_news_index.faiss',
    k: int = 3,
    max_new_tokens: int = 200
) -> str:
    """
    Args:
        query: Вопрос пользователя
        model: Языковая модель
        tokenizer: Токенизатор для модели
        sentence_transformer: Модель для эмбеддингов
        df: DataFrame с текстами (опционально)
        index_path: Путь к FAISS индексу
        k: Количество извлекаемых чанков
        max_new_tokens: Максимальная длина ответа
    """

    similarity, is_it = is_query_about_it(query, sentence_transformer)
    
    if not is_it:
        return get_off_topic_response()
    
    try:
        index = faiss.read_index(index_path)
        context_chunks = search_in_index(
            query, index, sentence_transformer, 
            df=df, k=k
        )
    except (FileNotFoundError, RuntimeError):
        context_chunks = []

    if not context_chunks:
        print('searching in internet...')
        texts = collect_info(query)
        preprocessed_texts = []
        
        for text in texts:
            preprocessed_texts.extend(
                preprocess_text_to_chunks(text['text'])
            )

        index = make_index(preprocessed_texts, sentence_transformer, index_path)
        context_chunks = search_in_index(
            query, index, sentence_transformer,
            texts=preprocessed_texts, k=k
        )

    context = "\n---\n".join(context_chunks) if context_chunks else "Нет релевантной информации в контексте."
    
    prompt = f"""You are an expert in technology and IT. You are provided with context from several news articles. 
        Use only this information to answer the user's question. 
        If the answer is not present in the context, honestly say that there is not enough information.

        News context:
        {context}

        User question:
        {query}

        Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    max_tokens = estimate_max_tokens(query, context)
    output_ids = model.generate(
    **inputs,
    max_new_tokens=max_tokens,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = response.split("Answer:")[-1].strip()
    
    return response


LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    LLM_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

model.eval()


EMB_MODEL = "all-MiniLM-L6-v2"

sentence_transformer = SentenceTransformer(
    EMB_MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

test_queries = [
    # 🔥 IT / AI
    "Which companies are leading in AI research?",
    "What is the difference between machine learning and deep learning?",
    "What are transformers used for in NLP?",

    # ☁️ IT / Cloud
    "What are the main cloud computing platforms?",
    "How does AWS differ from Google Cloud?",

    # 🔐 IT / Security
    "What is cybersecurity and why is it important?",

    # 🌐 Web fallback (нет в локальной базе)
    "What is Anthropic focusing on in AI safety?",
    "What are the latest trends in large language models?",

    # ❌ Не IT
    "What is the best diet for weight loss?",
    "Who won the last football World Cup?",
]

# import pandas as pd
# df = pd.read_csv('data/rag_data.csv')
# for i, query in enumerate(test_queries, 1):
#     print("=" * 80)
#     print(f"[{i}] USER QUERY:")
#     print(query)
#     answer = rag_agent(
#         query=query,
#         model=model,
#         tokenizer=tokenizer,
#         sentence_transformer=sentence_transformer,
#     df = df,
#     index_path= 'data/tech_news_index.faiss',
#     k= 5,
#     max_new_tokens = 200
#     )

#     print("\nASSISTANT ANSWER:")
#     print(answer)


print(rag_agent(
    query="What is cybersecurity?",
    model=model,
    tokenizer=tokenizer,
    sentence_transformer=sentence_transformer,
))