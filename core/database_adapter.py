# core/database_adapter.py
import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseAdapter:
    def __init__(self):
        self.engine = None
        self.db_type = None
        self.connect()
    
    def connect(self):
        """Connect to database (PostgreSQL or SQLite)"""
        try:
            # Try PostgreSQL first (Docker environment)
            postgres_url = os.getenv('DATABASE_URL')
            if postgres_url and postgres_url.startswith('postgresql'):
                self.engine = create_engine(postgres_url)
                self.db_type = 'postgresql'
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(f"Connected to PostgreSQL: {postgres_url}")
                return
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
        
        try:
            # Fallback to SQLite (legacy/development)
            sqlite_path = os.getenv('DB_PATH', 'data/system_data.db')
            sqlite_url = f'sqlite:///{sqlite_path}'
            self.engine = create_engine(sqlite_url)
            self.db_type = 'sqlite'
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Connected to SQLite: {sqlite_url}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.engine = None
            self.db_type = None
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        if self.engine is None:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
    
    def get_table_prefix(self) -> str:
        """Get table prefix based on database type"""
        if self.db_type == 'postgresql':
            return 'trading.'
        else:
            return ''
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> Optional[pd.DataFrame]:
        """Execute a query and return results as DataFrame"""
        if not self.is_connected():
            logger.error("Database not connected")
            return None
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn, params=params or {})
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
    
    def execute_statement(self, statement: str, params: Dict[str, Any] = None) -> bool:
        """Execute a statement (INSERT, UPDATE, DELETE)"""
        if not self.is_connected():
            logger.error("Database not connected")
            return False
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(statement), params or {})
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Statement execution failed: {e}")
            return False
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> bool:
        """Insert DataFrame into table"""
        if not self.is_connected() or df.empty:
            return False
        
        try:
            table_full_name = f"{self.get_table_prefix()}{table_name}"
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False, 
                     schema='trading' if self.db_type == 'postgresql' else None)
            logger.debug(f"Inserted {len(df)} rows into {table_full_name}")
            return True
        except Exception as e:
            logger.error(f"DataFrame insertion failed for {table_name}: {e}")
            return False
    
    def get_price_data(self, pair: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get price data for a specific pair"""
        table_prefix = self.get_table_prefix()
        query = f"""
        SELECT timestamp, open, high, low, close, volume 
        FROM {table_prefix}price_data 
        WHERE pair = :pair 
        ORDER BY timestamp DESC 
        LIMIT :limit
        """
        return self.execute_query(query, {'pair': pair, 'limit': limit})
    
    def get_news_articles(self, limit: int = 100, processed_only: bool = False) -> Optional[pd.DataFrame]:
        """Get news articles"""
        table_prefix = self.get_table_prefix()
        where_clause = ""
        if processed_only:
            where_clause = "WHERE processed_for_ai_trade_decision = 1"
        
        query = f"""
        SELECT id, title, content, published_at, source, processed_for_ai_trade_decision
        FROM {table_prefix}news_articles 
        {where_clause}
        ORDER BY published_at DESC 
        LIMIT :limit
        """
        return self.execute_query(query, {'limit': limit})
    
    def get_ai_trade_decisions(self, limit: int = 100, model_name: str = None) -> Optional[pd.DataFrame]:
        """Get AI trade decisions"""
        table_prefix = self.get_table_prefix()
        where_clause = ""
        params = {'limit': limit}
        
        if model_name:
            where_clause = "WHERE model_name = :model_name"
            params['model_name'] = model_name
        
        query = f"""
        SELECT * FROM {table_prefix}ai_trade_decisions 
        {where_clause}
        ORDER BY processed_at DESC 
        LIMIT :limit
        """
        return self.execute_query(query, params)
    
    def get_openai_usage(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OpenAI API usage statistics"""
        table_prefix = self.get_table_prefix()
        query = f"""
        SELECT timestamp, model_name, prompt_tokens, completion_tokens, 
               total_tokens, estimated_cost_usd, purpose
        FROM {table_prefix}openai_api_usage 
        ORDER BY timestamp DESC 
        LIMIT :limit
        """
        return self.execute_query(query, {'limit': limit})
    
    def insert_price_data(self, price_data: Dict[str, Any]) -> bool:
        """Insert single price data point"""
        table_prefix = self.get_table_prefix()
        
        if self.db_type == 'postgresql':
            query = f"""
            INSERT INTO {table_prefix}price_data 
            (timestamp, pair, open, high, low, close, volume)
            VALUES (:timestamp, :pair, :open, :high, :low, :close, :volume)
            """
        else:
            query = f"""
            INSERT OR REPLACE INTO {table_prefix}price_data 
            (timestamp, pair, open, high, low, close, volume)
            VALUES (:timestamp, :pair, :open, :high, :low, :close, :volume)
            """
        
        return self.execute_statement(query, price_data)
    
    def insert_news_article(self, article_data: Dict[str, Any]) -> bool:
        """Insert single news article"""
        table_prefix = self.get_table_prefix()
        
        if self.db_type == 'postgresql':
            query = f"""
            INSERT INTO {table_prefix}news_articles 
            (id, title, content, published_at, source, url)
            VALUES (:id, :title, :content, :published_at, :source, :url)
            ON CONFLICT (id) DO NOTHING
            """
        else:
            query = f"""
            INSERT OR IGNORE INTO {table_prefix}news_articles 
            (id, title, content, published_at, source, url)
            VALUES (:id, :title, :content, :published_at, :source, :url)
            """
        
        return self.execute_statement(query, article_data)
    
    def insert_ai_decision(self, decision_data: Dict[str, Any]) -> bool:
        """Insert AI trade decision"""
        table_prefix = self.get_table_prefix()
        
        if self.db_type == 'postgresql':
            query = f"""
            INSERT INTO {table_prefix}ai_trade_decisions 
            (news_id, processed_at, model_name, trade_warranted, pair, direction, 
             confidence, reasoning, stop_loss_pips, take_profit_pips, 
             suggested_lot_size_factor, raw_response_json)
            VALUES (:news_id, :processed_at, :model_name, :trade_warranted, :pair, 
                    :direction, :confidence, :reasoning, :stop_loss_pips, 
                    :take_profit_pips, :suggested_lot_size_factor, :raw_response_json)
            ON CONFLICT (news_id) DO NOTHING
            """
        else:
            query = f"""
            INSERT OR IGNORE INTO {table_prefix}ai_trade_decisions 
            (news_id, processed_at, model_name, trade_warranted, pair, direction, 
             confidence, reasoning, stop_loss_pips, take_profit_pips, 
             suggested_lot_size_factor, raw_response_json)
            VALUES (:news_id, :processed_at, :model_name, :trade_warranted, :pair, 
                    :direction, :confidence, :reasoning, :stop_loss_pips, 
                    :take_profit_pips, :suggested_lot_size_factor, :raw_response_json)
            """
        
        return self.execute_statement(query, decision_data)
    
    def insert_api_usage(self, usage_data: Dict[str, Any]) -> bool:
        """Insert OpenAI API usage record"""
        table_prefix = self.get_table_prefix()
        query = f"""
        INSERT INTO {table_prefix}openai_api_usage 
        (timestamp, model_name, prompt_tokens, completion_tokens, total_tokens, 
         estimated_cost_usd, related_news_id, purpose)
        VALUES (:timestamp, :model_name, :prompt_tokens, :completion_tokens, 
                :total_tokens, :estimated_cost_usd, :related_news_id, :purpose)
        """
        
        return self.execute_statement(query, usage_data)
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """Cleanup old data (for SQLite mainly, PostgreSQL has retention policies)"""
        if self.db_type == 'postgresql':
            # PostgreSQL has automatic retention policies via TimescaleDB
            return True
        
        try:
            table_prefix = self.get_table_prefix()
            cutoff_date = f"datetime('now', '-{days_to_keep} days')"
            
            # Cleanup old price data
            query1 = f"DELETE FROM {table_prefix}price_data WHERE timestamp < {cutoff_date}"
            self.execute_statement(query1)
            
            # Cleanup old news
            query2 = f"DELETE FROM {table_prefix}news_articles WHERE published_at < {cutoff_date}"
            self.execute_statement(query2)
            
            # Cleanup old API usage
            query3 = f"DELETE FROM {table_prefix}openai_api_usage WHERE timestamp < {cutoff_date}"
            self.execute_statement(query3)
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return False

# Global database adapter instance
db_adapter = DatabaseAdapter()