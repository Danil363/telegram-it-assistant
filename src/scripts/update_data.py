from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))  # Поднимаемся на уровень выше
import pandas as pd
import os
from ingestion.get_news import collect_news
from preprocessing.preprocess_text import preprocess_text_to_chunks



file_name = 'news_data.csv'

def update_data(file_name):
    f_path = os.path.join('data', file_name)

    if os.path.exists(f_path):
        df_old = pd.read_csv(f_path)
        df_new = collect_news(max_articles_per_feed=100)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset="url")
        df.to_csv(f_path)
    else:
        df = collect_news(max_articles_per_feed=20)
        df.to_csv(f_path)

    return df

def make_rag_data(df: pd.DataFrame):
    texts = df['text']
    urls = df['url']
    data = []
    f_path = os.path.join('data', 'rag_data.csv')

    for text, url in zip(texts, urls):
        chunks = preprocess_text_to_chunks(text)
        for chunk in chunks:
            data.append({'text':chunk, 'url':url})

    df = pd.DataFrame(data)
    df.to_csv(f_path, index=False)


if __name__ == '__main__':
    df = update_data(file_name)
    # df = pd.read_csv('data/news_data.csv')
    make_rag_data(df)
    print('Updating was sucecess')