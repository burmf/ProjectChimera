import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.ConnectionError))
)
def fetch_news():
    """ニュースAPIから為替関連記事を取得"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "USD JPY OR 為替 OR 外国為替",
        "sortBy": "publishedAt",
        "language": "ja",
        "apiKey": os.getenv("NEWS_API_KEY")
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        print("API response:", response.json())

        df = pd.DataFrame([{
            "timestamp": datetime.strptime(art["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"),
            "title": art["title"],
            "content": art["content"],
            "source": art["source"]["name"],
            "url": art["url"]
        } for art in articles])

        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(f"data/raw/news_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        print(f"取得完了: {len(df)}件のニュースを保存")

    except requests.exceptions.RequestException as e:
        print(f"リトライ上限に達しました: {str(e)}")

if __name__ == "__main__":
    fetch_news()