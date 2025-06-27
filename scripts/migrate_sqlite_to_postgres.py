import os
import sqlite3

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

# 環境変数の読み込み
load_dotenv()

# データベース接続情報
SQLITE_DB = 'data/system_data.db'
POSTGRES_URL = os.getenv('DATABASE_URL')

def connect_sqlite():
    return sqlite3.connect(SQLITE_DB)

def connect_postgres():
    return psycopg2.connect(POSTGRES_URL)

def migrate_table(sqlite_cur, pg_cur, table_name):
    print(f"Migrating table: {table_name}")

    # SQLiteからデータを取得
    sqlite_cur.execute(f"SELECT * FROM {table_name}")
    rows = sqlite_cur.fetchall()

    if not rows:
        print(f"No data found in {table_name}")
        return

    # カラム名を取得
    columns = [description[0] for description in sqlite_cur.description]

    # PostgreSQLにデータを挿入（tradingスキーマを追加）
    placeholders = ','.join(['%s'] * len(columns))
    columns_str = ','.join(columns)

    for row in rows:
        try:
            pg_cur.execute(
                f"INSERT INTO trading.{table_name} ({columns_str}) VALUES ({placeholders})",
                row
            )
        except Exception as e:
            print(f"Error inserting row in {table_name}: {e}")
            continue

def migrate_timeseries_table(sqlite_cur, pg_engine, table_name):
    print(f"Migrating timeseries table: {table_name}")

    # SQLiteからデータをPandasデータフレームとして読み込む
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_cur)

    if df.empty:
        print(f"No data found in {table_name}")
        return

    # タイムスタンプ列を変換
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # PostgreSQLに一括挿入（tradingスキーマを指定）
    try:
        df.to_sql(table_name, pg_engine, schema='trading', if_exists='append', index=False)
    except Exception as e:
        print(f"Error inserting data in {table_name}: {e}")

def main():
    # SQLite接続
    sqlite_conn = connect_sqlite()
    sqlite_cur = sqlite_conn.cursor()

    # PostgreSQL接続
    pg_conn = connect_postgres()
    pg_cur = pg_conn.cursor()
    pg_engine = create_engine(POSTGRES_URL)

    try:
        # 通常テーブルの移行
        regular_tables = [
            'news_articles',
            'news_tags',
            'trade_history',
            'ai_trade_decisions',
            'openai_manual_analysis'
        ]

        for table in regular_tables:
            migrate_table(sqlite_cur, pg_cur, table)

        # TimescaleDBテーブルの移行
        timeseries_tables = [
            'price_data',
            'alpha_signals',
            'openai_api_usage'
        ]

        for table in timeseries_tables:
            migrate_timeseries_table(sqlite_cur, pg_engine, table)

        # 変更をコミット
        pg_conn.commit()
        print("Migration completed successfully")

    except Exception as e:
        pg_conn.rollback()
        print(f"Migration failed: {e}")

    finally:
        # 接続をクローズ
        sqlite_cur.close()
        sqlite_conn.close()
        pg_cur.close()
        pg_conn.close()

if __name__ == '__main__':
    main()
