import os
import json
import openai
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_trading_hypothesis():
    """ニュースデータを分析してGPTが仮説生成"""
    latest_news = max([f for f in os.listdir("data/raw") if f.startswith("news_")], default=None)
    
    if not latest_news:
        print("分析対象のニュースデータが見つかりません")
        return

    df = pd.read_csv(f"data/raw/{latest_news}")
    news_text = "\n".join([f"{row['timestamp']} {row['title']}: {row['content']}" for _, row in df.iterrows()])

    prompt = f"""為替トレーダーとして以下のニュースを分析してください：
{news_text}

出力形式：
{{
  "hypothesis": "USD/JPYの方向性（上昇/下降/横ばい）",
  "confidence": 0-100の信頼度,
  "reason": "根拠となるニュースの要約",
  "time_horizon": "効果が持続すると予想する時間枠（1時間/1日/1週間）"
}}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7
        )
        
        hypothesis = json.loads(response.choices[0].message.content)
        output_path = f"data/processed/hypothesis_{datetime.now().strftime('%Y%m%d')}.json"
        
        os.makedirs("data/processed", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(hypothesis, f, indent=2, ensure_ascii=False)
            
        print(f"仮説生成完了: {output_path}")

    except Exception as e:
        print(f"生成エラー: {str(e)}")

if __name__ == "__main__":
    generate_trading_hypothesis()