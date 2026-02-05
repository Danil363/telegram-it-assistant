import feedparser
import pandas as pd
from newspaper import Article
from langdetect import detect
from datetime import datetime
import re

RSS_FEEDS = {
    "techcrunch": "https://techcrunch.com/feed/",
    "verge": "https://www.theverge.com/rss/index.xml",
    "wired": "https://www.wired.com/feed/rss",
    "reuters-tech": "https://www.reuters.com/rssFeed/technologyNews"
}

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    return text


def fetch_article(url: str) -> dict | None:
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = clean_text(article.text)
        if len(text) < 200:
            return None

        lang = detect(text)

        return {
            "headline": article.title,
            "text": text,
            "language": lang,
        }

    except Exception as e:
        return None


def collect_news(rss_feeds: dict = RSS_FEEDS, max_articles_per_feed: int = 10) -> pd.DataFrame:
    rows = []

    for source, feed_url in rss_feeds.items():
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:max_articles_per_feed]:
            article_data = fetch_article(entry.link)

            if not article_data:
                continue

            if article_data["language"] != "en":
                continue

            rows.append({
                "source": source,
                "url": entry.link,
                "headline": article_data["headline"],
                "text": article_data["text"],
                "published": entry.get("published", None),
                "collected_at": datetime.utcnow()
            })

    return pd.DataFrame(rows)

