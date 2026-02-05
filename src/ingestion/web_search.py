import urllib.parse
import requests
from bs4 import BeautifulSoup
import trafilatura
from serpapi import GoogleSearch

SERPAPI_KEY = "9a4809cff1b0e9a3e15243002cd3ad0842b5db33077e99fe5fc69f0803070a6a"  
MAX_RESULTS = 5


def search_serpapi(query, max_results=MAX_RESULTS):
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": SERPAPI_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    urls = []

    if "organic_results" in results:
        for r in results["organic_results"]:
            if "link" in r:
                urls.append({"title": r.get("title", ""), "url": r["link"]})
    return urls

def extract_text(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    text = trafilatura.extract(downloaded)
    return text or ""

def collect_info(query):
    urls = search_serpapi(query)
    collected_texts = []

    for u in urls:
        print(f"Processing: {u['title']}")
        text = extract_text(u["url"])
        if text:
            collected_texts.append({"title": u["title"], "url": u["url"], "text": text})
        else:
            print("No text found:", u["url"])

    return collected_texts


# query = "Which companies are leading in AI research?"
# data = collect_info(query)

# for item in data:
#     print("TITLE:", item["title"])
#     print("URL:", item["url"])
#     print("TEXT LENGTH:", len(item["text"]))
#     print("-" * 80)
