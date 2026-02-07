from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM,  AutoTokenizer
import torch

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

API_TOKEN = "API_TOKEN "
DATA_PATH = 'data/rag_data.csv'
INDEX_PATH = 'data/tech_news_index.faiss'
