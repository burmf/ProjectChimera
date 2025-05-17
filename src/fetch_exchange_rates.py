import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def fetch_fx_rates():
    """Alpha Vantageから為替レートを取得"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": "USD",
        "to_symbol": "JPY",
        "apikey": os.getenv("ALPHA_VANTAGE_KEY"),
        "datatype": "csv"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(f"data/raw/fx_rates_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        print(f"取得完了: {len(df)}日分の為替データを保存")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    fetch_fx_rates()