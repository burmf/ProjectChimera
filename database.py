# database.py
import sqlite3

DB_PATH = 'data/system_data.db'  # データベースファイルの保存場所

def create_connection():
    """データベースへの接続を作成する"""
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        print(f"データベース接続エラー: {e}")
    return None

def setup_database():
    """システムの全てのテーブルを初期化（作成）する関数"""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                timestamp TEXT PRIMARY KEY,
                pair TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER
            )''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                published_at TEXT NOT NULL,
                source TEXT,
                title TEXT,
                content TEXT,
                url TEXT UNIQUE,
                processed_for_ai_trade_decision INTEGER DEFAULT 0
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_id TEXT NOT NULL, 
                tag_type TEXT NOT NULL,
                tag_value TEXT NOT NULL,
                FOREIGN KEY (news_id) REFERENCES news_articles (id)
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alpha_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, 
                source_module TEXT NOT NULL, 
                asset TEXT NOT NULL,
                signal_value REAL, 
                confidence REAL, 
                metadata TEXT
            )''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                lot_size REAL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl_pips REAL,
                pnl_currency REAL,
                stop_loss_price REAL,
                take_profit_price REAL,
                entry_reason TEXT,
                exit_reason TEXT,
                triggered_by_news_id TEXT,
                notes TEXT
            )''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_trade_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_id TEXT NOT NULL UNIQUE,
                processed_at TEXT NOT NULL,
                model_name TEXT NOT NULL,
                trade_warranted INTEGER NOT NULL,
                pair TEXT,
                direction TEXT,
                confidence REAL,
                reasoning TEXT,
                stop_loss_pips INTEGER,
                take_profit_pips INTEGER,
                suggested_lot_size_factor REAL,
                raw_response_json TEXT,
                FOREIGN KEY (news_id) REFERENCES news_articles (id)
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS openai_api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                estimated_cost_usd REAL,
                related_news_id TEXT,
                purpose TEXT
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS openai_manual_analysis (
                timestamp TEXT,
                model_name TEXT,
                news_input TEXT,
                system_prompt TEXT,
                response_json TEXT
            )''')

            print("データベースと全テーブルの準備が完了しました。")

        except sqlite3.Error as e:
            print(f"データベース設定中にエラーが発生しました: {e}")
        finally:
            conn.close()
    else:
        print("エラー：データベースに接続できません。")

if __name__ == '__main__':
    setup_database()