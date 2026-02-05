from pathlib import Path
import sys
import random
from typing import List, Tuple, Optional, Union
import asyncio

sys.path.append(str(Path(__file__).parent.parent))
from ingestion.web_search import collect_info
from preprocessing.preprocess_text import preprocess_text_to_chunks

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from transformers import AutoModelForCausalLM,  AutoTokenizer
import pandas as pd
import torch

class RAGAgent:
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

    GENERATING_RESPONSE_MESSAGES = [
    "⏳ It may take some time to generate a response...",
    "⏳ Generating your answer, please wait...",
    "⏳ This might take a few seconds, hang tight...",
    "⏳ Processing your request, please be patient...",
    "⏳ Formulating a detailed response, one moment...",
    "⏳ Your answer is being prepared, please wait...",
    "⏳ Working on your request, almost done...",
    "⏳ Hold on, generating your response now...",
    "⏳ Preparing an answer, this may take a moment...",
    "⏳ Generating a thoughtful response for you..."
]
    
    IT_SIMILARITY_THRESHOLD = 0.25
    SEARCH_SIMILARITY_THRESHOLD = 0.45
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        sentence_transformer: SentenceTransformer,
        df_path: Optional[str] = None,
        index_path: str = 'data/tech_news_index.faiss',
        device: str = None
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.sentence_transformer = sentence_transformer
        self.df = pd.read_csv(df_path)
        self.df_path =df_path
        self.index_path = index_path
        self.index = None
        self.texts = None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._load_index()
        
        self.it_topic_embedding = self.sentence_transformer.encode(self.IT_TOPIC)
    
    def _load_index(self) -> None:
        try:
            self.index = faiss.read_index(self.index_path)
            print(f"Индекс успешно загружен из {self.index_path}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Не удалось загрузить индекс: {e}")
            self.index = None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def is_query_about_it(self, query: str, threshold: float = None) -> Tuple[float, bool]:
        if threshold is None:
            threshold = self.IT_SIMILARITY_THRESHOLD
        
        query_embedding = self.sentence_transformer.encode(query)
        similarity = self._cosine_similarity(self.it_topic_embedding, query_embedding)

        return similarity, similarity >= threshold
    
    def create_index(
        self,
        texts: List[str],
        save_path: Optional[str] = None
    ) -> faiss.Index:

        if not texts:
            raise ValueError("No texts to index")
        
        print(f"Создание индекса для {len(texts)} текстов...")
        embeddings = self.sentence_transformer.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        faiss.normalize_L2(embeddings)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        

        # if save_path:
        #     faiss.write_index(index, save_path)
        #     print(f"Индекс сохранен в {save_path}")
        

        return index, embeddings
    
    def search_in_index(
        self,
        query: str,
        index: faiss.Index,
        *,
        texts: List[str] = None,
        df = None,
        k: int = 3,
        similarity_threshold: float = SEARCH_SIMILARITY_THRESHOLD
    ) -> List[str]:

        q_emb = self.sentence_transformer.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        
        D, I = index.search(q_emb, k)
        context_chunks = []
        urls = set()
        
        for idx, score in zip(I[0], D[0]):
            if score > similarity_threshold:
                if df is not None:
                    context_chunks.append(df.iloc[idx]['text'])
                    urls.add(df.iloc[idx]['url'])
                elif texts is not None:
                    context_chunks.append(texts[idx])
        
        return context_chunks, urls

    def _get_off_topic_response(self) -> str:
        return random.choice(self.OFF_TOPIC_RESPONSES)
    
    def _construct_prompt(self, query: str, context: str) -> str:
        prompt = f"""You are an expert in technology and IT. You are provided with context from several news articles. 
                    Use only this information to answer the user's question. 
                    If the answer is not present in the context, honestly say that there is not enough information.

                    News context:
                    {context}

                    User question:
                    {query}

                    Answer:"""
        return prompt

    def _generate_response(self, prompt: str, max_tokens: int) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    
    async def generate_answer(
        self,
        query: str,
        user_message,          
        max_tokens: int = 128,
        k: int = 3,
        search_online: bool = True,
        detail: bool = False
    ) -> Tuple[str, list]:

        similarity, is_it = self.is_query_about_it(query)
        
        if not is_it and not detail:
            print(f"The request is not related to IT")
            return self._get_off_topic_response(), []

        print(f"Запрос относится к IT (схожесть: {similarity:.3f})")
        
        context_chunks, urls = self.search_in_index(query, self.index, df=self.df, k=k)
        
        if not context_chunks and search_online:
            await user_message.answer("🔎 I'm searching for information online, this may take a few seconds...")
            print('searching in internet...')
            
            texts = collect_info(query)
            preprocessed_texts = []
            urls = []

            for text in texts:
                chunks = preprocess_text_to_chunks(text['text'])
                preprocessed_texts.extend(chunks)
                urls.extend([text['url']] * len(chunks))

            index, embedings = self.create_index(preprocessed_texts, self.sentence_transformer)
            context_chunks, _ = self.search_in_index(
                query, 
                index, 
                texts=preprocessed_texts,
                k=k
            )
            await (self._update_index(self.index, embedings, preprocessed_texts, urls, self.df))

        if context_chunks:
            context = "\n---\n".join(context_chunks)
            print(f"Найдено {len(context_chunks)} релевантных фрагментов")
        else:
            context = "No relevant information in context."
            print("Релевантные фрагменты не найдены")

        await user_message.answer(random.choice(self.GENERATING_RESPONSE_MESSAGES))
        prompt = self._construct_prompt(query, context)
                
        print(f"Генерация ответа (макс. токенов: {max_tokens})...")
        
        response = self._generate_response(prompt, max_tokens)
        print(set(urls))
        
        return response, set(urls)


    async def _update_index(self, index, embeddings, new_texts, new_urls, df_):
        new_df = pd.DataFrame({'text': new_texts, 'url': new_urls})
        combined_df = pd.concat([df_, new_df], ignore_index=False)

        index.add(embeddings)

        await asyncio.to_thread(faiss.write_index, index, self.index_path)
        await asyncio.to_thread(combined_df.to_csv, self.df_path, index=False)

        self.df = combined_df
        



# LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(
#     LLM_NAME,
#     trust_remote_code=True
# )

# model = AutoModelForCausalLM.from_pretrained(
#     LLM_NAME,
#     dtype=torch.float16,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# )
# tokenizer.pad_token = tokenizer.eos_token

# model.eval()


# EMB_MODEL = "all-MiniLM-L6-v2"

# sentence_transformer = SentenceTransformer(
#     EMB_MODEL,
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )

# test_queries = [
#     # 🔥 IT / AI
#     "Which companies are leading in AI research?",
#     "What is the difference between machine learning and deep learning?",
#     "What are transformers used for in NLP?",

#     # ☁️ IT / Cloud
#     "What are the main cloud computing platforms?",
#     "How does AWS differ from Google Cloud?",

#     # 🔐 IT / Security
#     "What is cybersecurity and why is it important?",

#     # 🌐 Web fallback (нет в локальной базе)
#     "What is Anthropic focusing on in AI safety?",
#     "What are the latest trends in large language models?",

#     # ❌ Не IT
#     "What is the best diet for weight loss?",
#     "Who won the last football World Cup?",
# ]

# # import pandas as pd
# # df = pd.read_csv('data/rag_data.csv')
# # for i, query in enumerate(test_queries, 1):
# #     print("=" * 80)
# #     print(f"[{i}] USER QUERY:")
# #     print(query)
# #     answer = rag_agent(
# #         query=query,
# #         model=model,
# #         tokenizer=tokenizer,
# #         sentence_transformer=sentence_transformer,
# #     df = df,
# #     index_path= 'data/tech_news_index.faiss',
# #     k= 5,
# #     max_new_tokens = 200
# #     )

# #     print("\nASSISTANT ANSWER:")
# #     print(answer)

# agent = RAGAgent(model, tokenizer, sentence_transformer, 'data/rag_data.csv')
# print(agent.generate_answer(
#     query="What is cybersecurity?"
# ))