#!/usr/bin/env python3
"""
Sentiment Analysis AI Department
ニュース・センチメント分析部門AI
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os
from collections import Counter
import math


from core.ai_agent_base import AIAgentBase, DepartmentType, AnalysisRequest, AnalysisResult
from core.department_prompts import DepartmentPrompts, PromptFormatter


class SentimentAnalysisAI(AIAgentBase):
    """
    ニュース・センチメント分析専門AIエージェント
    
    市場センチメント、ニュースインパクト、投資家心理の分析を専門とする
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(DepartmentType.SENTIMENT, model_config)
        
        # センチメント分析用の辞書
        self.sentiment_lexicon = self._initialize_sentiment_lexicon()
        
        # ニュースソース別の信頼度重み
        self.source_weights = {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'financial times': 0.9,
            'wall street journal': 0.9,
            'cnbc': 0.8,
            'marketwatch': 0.7,
            'yahoo finance': 0.6,
            'investing.com': 0.6,
            'default': 0.5
        }
        
        # センチメント閾値設定
        self.sentiment_thresholds = {
            'very_positive': 0.6,
            'positive': 0.2,
            'neutral': 0.1,
            'negative': -0.2,
            'very_negative': -0.6
        }
        
        # 市場インパクトレベルの設定
        self.impact_factors = {
            'breaking_news': 2.0,
            'central_bank': 1.8,
            'employment': 1.5,
            'inflation': 1.5,
            'gdp': 1.3,
            'trade': 1.2,
            'geopolitical': 1.4,
            'earnings': 1.0,
            'technical': 0.8
        }
        
        # 感情的バイアス検出パターン
        self.bias_patterns = {
            'fear': ['crash', 'collapse', 'panic', 'crisis', 'disaster', 'plunge'],
            'greed': ['surge', 'boom', 'rally', 'soar', 'skyrocket', 'explosive'],
            'uncertainty': ['unclear', 'volatile', 'mixed', 'conflicting', 'uncertain']
        }
        
        self.logger.info("Sentiment Analysis AI initialized")
    
    def _initialize_sentiment_lexicon(self) -> Dict[str, float]:
        """センチメント分析用語辞書の初期化"""
        return {
            # 強いポジティブ (0.8-1.0)
            'surge': 0.9, 'soar': 0.9, 'rally': 0.8, 'boom': 0.9, 'breakthrough': 0.8,
            'optimistic': 0.8, 'bullish': 0.9, 'confident': 0.8, 'strong': 0.7,
            'growth': 0.7, 'gain': 0.6, 'rise': 0.6, 'improve': 0.7, 'positive': 0.6,
            
            # 中程度ポジティブ (0.3-0.7)
            'increase': 0.5, 'advance': 0.5, 'progress': 0.6, 'recover': 0.6,
            'stable': 0.4, 'steady': 0.4, 'solid': 0.5, 'healthy': 0.6,
            
            # 中立 (-0.2-0.2)
            'neutral': 0.0, 'unchanged': 0.0, 'flat': 0.0, 'mixed': 0.0,
            
            # 中程度ネガティブ (-0.7--0.3)
            'decline': -0.5, 'fall': -0.5, 'drop': -0.5, 'decrease': -0.4,
            'concern': -0.4, 'worry': -0.5, 'weak': -0.5, 'slow': -0.4,
            'pressure': -0.4, 'challenge': -0.3, 'risk': -0.4,
            
            # 強いネガティブ (-1.0--0.8)
            'crash': -0.9, 'collapse': -0.9, 'plunge': -0.8, 'crisis': -0.8,
            'panic': -0.9, 'disaster': -0.9, 'bearish': -0.8, 'recession': -0.8,
            'uncertainty': -0.6, 'volatile': -0.5, 'turmoil': -0.7
        }
    
    def _get_system_prompt(self) -> str:
        """センチメント分析専用システムプロンプトを取得"""
        return DepartmentPrompts.get_system_prompt(DepartmentType.SENTIMENT)
    
    def _validate_request_data(self, request: AnalysisRequest) -> bool:
        """リクエストデータの妥当性チェック"""
        try:
            data = request.data
            
            # ニュースデータの存在確認
            if 'news_data' not in data:
                self.logger.error("News data is required for sentiment analysis")
                return False
            
            news_data = data['news_data']
            if not isinstance(news_data, list):
                self.logger.error("News data must be a list")
                return False
            
            if len(news_data) == 0:
                self.logger.warning("Empty news data provided")
                return True  # 空データでも処理続行
            
            # ニュース項目の基本構造チェック
            for news in news_data[:3]:  # 最初の3件をチェック
                if not isinstance(news, dict):
                    self.logger.error("Each news item must be a dictionary")
                    return False
                
                if 'title' not in news and 'content' not in news:
                    self.logger.error("News items must have title or content")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
    
    async def _analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """センチメント分析のメインロジック"""
        try:
            data = request.data
            news_data = data.get('news_data', [])
            price_data = data.get('price_data', {})
            
            if not news_data:
                return self._generate_neutral_sentiment_result()
            
            # ニュース前処理とフィルタリング
            processed_news = self._preprocess_news_data(news_data)
            
            # 個別ニュースのセンチメント分析
            news_sentiments = self._analyze_individual_news(processed_news)
            
            # 全体的なセンチメントスコア計算
            overall_sentiment = self._calculate_overall_sentiment(news_sentiments)
            
            # センチメントトレンドの分析
            sentiment_trend = self._analyze_sentiment_trend(news_sentiments)
            
            # 市場インパクトレベルの評価
            market_impact = self._evaluate_market_impact(news_sentiments, processed_news)
            
            # 投資家行動パターンの分析
            investor_behavior = self._analyze_investor_behavior(news_sentiments, overall_sentiment)
            
            # ニュースの新鮮度とタイミング分析
            timing_analysis = self._analyze_timing_factors(processed_news)
            
            # 逆張りシグナルの検出
            contrarian_signals = self._detect_contrarian_signals(
                overall_sentiment, investor_behavior, market_impact
            )
            
            # 総合センチメント判定
            final_assessment = self._generate_sentiment_decision(
                overall_sentiment,
                sentiment_trend,
                market_impact,
                investor_behavior,
                timing_analysis,
                contrarian_signals
            )
            
            return {
                'sentiment_score': overall_sentiment['score'],
                'sentiment_trend': sentiment_trend['direction'],
                'news_impact_level': market_impact['level'],
                'market_mood': investor_behavior['market_mood'],
                'key_sentiment_drivers': overall_sentiment['key_drivers'],
                'news_analysis': {
                    'positive_factors': overall_sentiment['positive_factors'],
                    'negative_factors': overall_sentiment['negative_factors'],
                    'neutral_factors': overall_sentiment['neutral_factors']
                },
                'investor_behavior': investor_behavior,
                'timing_analysis': timing_analysis,
                'action': final_assessment['action'],
                'confidence': final_assessment['confidence'],
                'reasoning': final_assessment['reasoning'],
                'key_factors': final_assessment['key_factors'],
                'contrarian_signals': contrarian_signals
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {
                'error': str(e),
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f'センチメント分析エラー: {str(e)}'
            }
    
    def _preprocess_news_data(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ニュースデータの前処理"""
        processed_news = []
        
        for news in news_data:
            try:
                processed_item = {
                    'title': news.get('title', ''),
                    'content': news.get('content', ''),
                    'source': self._normalize_source(news.get('source', '')),
                    'published_at': news.get('published_at', ''),
                    'url': news.get('url', ''),
                    'original': news
                }
                
                # テキストのクリーニング
                processed_item['cleaned_text'] = self._clean_text(
                    processed_item['title'] + ' ' + processed_item['content']
                )
                
                # ニュースカテゴリの分類
                processed_item['category'] = self._categorize_news(processed_item)
                
                # 時間的新鮮度の計算
                processed_item['freshness'] = self._calculate_news_freshness(
                    processed_item['published_at']
                )
                
                # ソース信頼度の割り当て
                processed_item['source_weight'] = self._get_source_weight(
                    processed_item['source']
                )
                
                processed_news.append(processed_item)
                
            except Exception as e:
                self.logger.warning(f"Failed to process news item: {e}")
                continue
        
        return processed_news
    
    def _normalize_source(self, source: str) -> str:
        """ニュースソース名の正規化"""
        source_lower = source.lower().strip()
        
        # 主要ソースの正規化
        source_mapping = {
            'reuters': 'reuters',
            'bloomberg': 'bloomberg',
            'ft.com': 'financial times',
            'financial times': 'financial times',
            'wsj': 'wall street journal',
            'wall street journal': 'wall street journal',
            'cnbc': 'cnbc',
            'marketwatch': 'marketwatch',
            'yahoo': 'yahoo finance'
        }
        
        for key, normalized in source_mapping.items():
            if key in source_lower:
                return normalized
        
        return source_lower
    
    def _clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        if not text:
            return ""
        
        # HTMLタグの除去
        text = re.sub(r'<[^>]+>', '', text)
        
        # 特殊文字の正規化
        text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', text)
        
        # 複数の空白を単一の空白に
        text = re.sub(r'\s+', ' ', text)
        
        # 先頭・末尾の空白除去
        text = text.strip()
        
        return text.lower()
    
    def _categorize_news(self, news_item: Dict[str, Any]) -> str:
        """ニュースのカテゴリ分類"""
        text = news_item['cleaned_text']
        
        category_keywords = {
            'central_bank': ['federal reserve', 'fed', 'central bank', 'monetary policy', 'interest rate'],
            'employment': ['employment', 'unemployment', 'job', 'payroll', 'labor'],
            'inflation': ['inflation', 'cpi', 'ppi', 'price pressure'],
            'gdp': ['gdp', 'economic growth', 'recession', 'expansion'],
            'trade': ['trade', 'tariff', 'export', 'import', 'trade war'],
            'geopolitical': ['war', 'conflict', 'sanctions', 'election', 'political'],
            'earnings': ['earnings', 'profit', 'revenue', 'quarterly results'],
            'technical': ['chart', 'resistance', 'support', 'breakout'],
            'market_structure': ['volatility', 'liquidity', 'volume', 'trading']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def _calculate_news_freshness(self, published_at: str) -> str:
        """ニュースの新鮮度計算"""
        try:
            if not published_at:
                return 'unknown'
            
            # 簡単な時間解析（ISO形式を想定）
            published_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            current_time = datetime.now()
            
            time_diff = current_time - published_time
            hours_diff = time_diff.total_seconds() / 3600
            
            if hours_diff < 1:
                return 'breaking'
            elif hours_diff < 6:
                return 'recent'
            elif hours_diff < 24:
                return 'today'
            elif hours_diff < 48:
                return 'yesterday'
            else:
                return 'stale'
                
        except Exception:
            return 'unknown'
    
    def _get_source_weight(self, source: str) -> float:
        """ソース信頼度重みの取得"""
        return self.source_weights.get(source, self.source_weights['default'])
    
    def _analyze_individual_news(
        self, 
        processed_news: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """個別ニュースのセンチメント分析"""
        news_sentiments = []
        
        for news in processed_news:
            try:
                sentiment_analysis = {
                    'news_item': news,
                    'raw_sentiment': self._calculate_lexicon_sentiment(news['cleaned_text']),
                    'weighted_sentiment': 0.0,
                    'impact_multiplier': 1.0,
                    'emotional_bias': self._detect_emotional_bias(news['cleaned_text']),
                    'market_relevance': self._assess_market_relevance(news)
                }
                
                # カテゴリ別インパクト倍率
                category = news['category']
                sentiment_analysis['impact_multiplier'] = self.impact_factors.get(category, 1.0)
                
                # 新鮮度による調整
                freshness_multiplier = self._get_freshness_multiplier(news['freshness'])
                
                # ソース信頼度による調整
                source_weight = news['source_weight']
                
                # 最終的な重み付きセンチメント
                sentiment_analysis['weighted_sentiment'] = (
                    sentiment_analysis['raw_sentiment'] *
                    sentiment_analysis['impact_multiplier'] *
                    freshness_multiplier *
                    source_weight
                )
                
                # センチメント強度の分類
                sentiment_analysis['intensity'] = self._classify_sentiment_intensity(
                    sentiment_analysis['weighted_sentiment']
                )
                
                news_sentiments.append(sentiment_analysis)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze news sentiment: {e}")
                continue
        
        return news_sentiments
    
    def _calculate_lexicon_sentiment(self, text: str) -> float:
        """辞書ベースのセンチメント計算"""
        if not text:
            return 0.0
        
        words = text.split()
        sentiment_scores = []
        
        for word in words:
            # 完全一致
            if word in self.sentiment_lexicon:
                sentiment_scores.append(self.sentiment_lexicon[word])
            else:
                # 部分一致（語幹など）
                for lex_word, score in self.sentiment_lexicon.items():
                    if lex_word in word or word in lex_word:
                        sentiment_scores.append(score * 0.7)  # 部分一致は重みを下げる
                        break
        
        if not sentiment_scores:
            return 0.0
        
        # 平均センチメント
        return sum(sentiment_scores) / len(sentiment_scores)
    
    def _detect_emotional_bias(self, text: str) -> Dict[str, float]:
        """感情的バイアスの検出"""
        bias_scores = {}
        
        for bias_type, keywords in self.bias_patterns.items():
            count = sum(1 for keyword in keywords if keyword in text)
            bias_scores[bias_type] = min(count / 10.0, 1.0)  # 正規化
        
        return bias_scores
    
    def _assess_market_relevance(self, news: Dict[str, Any]) -> float:
        """市場関連性の評価"""
        text = news['cleaned_text']
        
        # 通貨・FX関連キーワード
        fx_keywords = [
            'dollar', 'usd', 'yen', 'jpy', 'euro', 'eur', 'pound', 'gbp',
            'currency', 'forex', 'exchange rate', 'fx'
        ]
        
        # 経済・金融関連キーワード
        economic_keywords = [
            'economy', 'economic', 'finance', 'financial', 'market', 'trading',
            'investment', 'investor', 'stock', 'bond', 'commodity'
        ]
        
        fx_relevance = sum(1 for keyword in fx_keywords if keyword in text)
        econ_relevance = sum(1 for keyword in economic_keywords if keyword in text)
        
        total_relevance = fx_relevance * 2 + econ_relevance  # FX関連により高い重み
        
        return min(total_relevance / 10.0, 1.0)  # 正規化
    
    def _get_freshness_multiplier(self, freshness: str) -> float:
        """新鮮度による倍率"""
        multipliers = {
            'breaking': 1.5,
            'recent': 1.2,
            'today': 1.0,
            'yesterday': 0.7,
            'stale': 0.3,
            'unknown': 0.5
        }
        return multipliers.get(freshness, 0.5)
    
    def _classify_sentiment_intensity(self, sentiment_score: float) -> str:
        """センチメント強度の分類"""
        if sentiment_score >= self.sentiment_thresholds['very_positive']:
            return 'very_positive'
        elif sentiment_score >= self.sentiment_thresholds['positive']:
            return 'positive'
        elif sentiment_score >= self.sentiment_thresholds['neutral']:
            return 'neutral'
        elif sentiment_score >= self.sentiment_thresholds['negative']:
            return 'negative'
        else:
            return 'very_negative'
    
    def _calculate_overall_sentiment(
        self, 
        news_sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """全体的なセンチメントスコア計算"""
        if not news_sentiments:
            return {
                'score': 0.0,
                'key_drivers': [],
                'positive_factors': [],
                'negative_factors': [],
                'neutral_factors': []
            }
        
        # 重み付きセンチメントの集計
        weighted_scores = []
        positive_factors = []
        negative_factors = []
        neutral_factors = []
        
        for sentiment in news_sentiments:
            score = sentiment['weighted_sentiment']
            weighted_scores.append(score)
            
            news_title = sentiment['news_item']['title'][:50] + "..."
            
            if score > 0.2:
                positive_factors.append({
                    'title': news_title,
                    'score': score,
                    'category': sentiment['news_item']['category']
                })
            elif score < -0.2:
                negative_factors.append({
                    'title': news_title,
                    'score': score,
                    'category': sentiment['news_item']['category']
                })
            else:
                neutral_factors.append({
                    'title': news_title,
                    'score': score,
                    'category': sentiment['news_item']['category']
                })
        
        # 全体スコア（重み付き平均）
        overall_score = sum(weighted_scores) / len(weighted_scores)
        
        # 主要ドライバーの特定
        key_drivers = self._identify_key_sentiment_drivers(news_sentiments)
        
        return {
            'score': float(overall_score),
            'key_drivers': key_drivers,
            'positive_factors': sorted(positive_factors, key=lambda x: x['score'], reverse=True)[:5],
            'negative_factors': sorted(negative_factors, key=lambda x: x['score'])[:5],
            'neutral_factors': neutral_factors[:3]
        }
    
    def _identify_key_sentiment_drivers(
        self, 
        news_sentiments: List[Dict[str, Any]]
    ) -> List[str]:
        """主要センチメントドライバーの特定"""
        category_impacts = {}
        
        for sentiment in news_sentiments:
            category = sentiment['news_item']['category']
            score = abs(sentiment['weighted_sentiment'])
            
            if category not in category_impacts:
                category_impacts[category] = []
            category_impacts[category].append(score)
        
        # カテゴリ別の平均インパクト
        category_averages = {
            cat: sum(scores) / len(scores) 
            for cat, scores in category_impacts.items()
        }
        
        # 上位3つのドライバー
        top_drivers = sorted(
            category_averages.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return [driver[0] for driver in top_drivers]
    
    def _analyze_sentiment_trend(
        self, 
        news_sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """センチメントトレンドの分析"""
        try:
            if len(news_sentiments) < 3:
                return {'direction': 'stable', 'momentum': 0.0}
            
            # 時系列順にソート（新しい順）
            sorted_sentiments = sorted(
                news_sentiments,
                key=lambda x: x['news_item']['published_at'],
                reverse=True
            )
            
            # 直近、中間、過去のセンチメント
            recent_scores = [s['weighted_sentiment'] for s in sorted_sentiments[:3]]
            middle_scores = [s['weighted_sentiment'] for s in sorted_sentiments[3:6]] if len(sorted_sentiments) > 3 else recent_scores
            older_scores = [s['weighted_sentiment'] for s in sorted_sentiments[6:]] if len(sorted_sentiments) > 6 else middle_scores
            
            recent_avg = sum(recent_scores) / len(recent_scores)
            middle_avg = sum(middle_scores) / len(middle_scores) if middle_scores else recent_avg
            older_avg = sum(older_scores) / len(older_scores) if older_scores else middle_avg
            
            # トレンド計算
            recent_trend = recent_avg - middle_avg
            longer_trend = middle_avg - older_avg
            
            # 方向性の判定
            if recent_trend > 0.1 and longer_trend > 0.05:
                direction = 'improving'
            elif recent_trend < -0.1 and longer_trend < -0.05:
                direction = 'deteriorating'
            else:
                direction = 'stable'
            
            # モメンタム（変化の強度）
            momentum = abs(recent_trend) + abs(longer_trend) * 0.5
            
            return {
                'direction': direction,
                'momentum': float(momentum),
                'recent_change': float(recent_trend),
                'longer_change': float(longer_trend)
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment trend analysis failed: {e}")
            return {'direction': 'stable', 'momentum': 0.0}
    
    def _evaluate_market_impact(
        self, 
        news_sentiments: List[Dict[str, Any]], 
        processed_news: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """市場インパクトレベルの評価"""
        try:
            if not news_sentiments:
                return {'level': 1, 'factors': []}
            
            impact_factors = []
            total_impact = 0.0
            
            for sentiment in news_sentiments:
                news_item = sentiment['news_item']
                
                # 基本インパクト
                base_impact = abs(sentiment['weighted_sentiment'])
                
                # カテゴリ別追加インパクト
                category_impact = sentiment['impact_multiplier']
                
                # 新鮮度による追加インパクト
                freshness = news_item['freshness']
                if freshness == 'breaking':
                    freshness_bonus = 1.0
                elif freshness == 'recent':
                    freshness_bonus = 0.5
                else:
                    freshness_bonus = 0.0
                
                # 市場関連性による調整
                relevance = sentiment['market_relevance']
                
                news_impact = base_impact * category_impact * (1 + freshness_bonus) * relevance
                total_impact += news_impact
                
                if news_impact > 0.5:  # 高インパクトニュース
                    impact_factors.append({
                        'title': news_item['title'][:50] + "...",
                        'category': news_item['category'],
                        'impact_score': news_impact
                    })
            
            # 平均インパクト
            avg_impact = total_impact / len(news_sentiments)
            
            # レベル分類（1-5）
            if avg_impact > 2.0:
                level = 5
            elif avg_impact > 1.5:
                level = 4
            elif avg_impact > 1.0:
                level = 3
            elif avg_impact > 0.5:
                level = 2
            else:
                level = 1
            
            return {
                'level': level,
                'total_impact': float(total_impact),
                'average_impact': float(avg_impact),
                'high_impact_factors': sorted(impact_factors, key=lambda x: x['impact_score'], reverse=True)[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Market impact evaluation failed: {e}")
            return {'level': 1, 'factors': []}
    
    def _analyze_investor_behavior(
        self, 
        news_sentiments: List[Dict[str, Any]], 
        overall_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """投資家行動パターンの分析"""
        try:
            sentiment_score = overall_sentiment['score']
            
            # Fear & Greed Index（簡易版）
            fear_greed_index = self._calculate_fear_greed_index(news_sentiments)
            
            # 不確実性レベル
            uncertainty_level = self._assess_uncertainty_level(news_sentiments)
            
            # 群集行動の強度
            herd_behavior = self._assess_herd_behavior(news_sentiments)
            
            # 市場ムードの判定
            market_mood = self._determine_market_mood(
                sentiment_score, fear_greed_index, uncertainty_level
            )
            
            return {
                'fear_greed_index': fear_greed_index,
                'uncertainty_level': uncertainty_level,
                'herd_behavior': herd_behavior,
                'market_mood': market_mood,
                'behavioral_bias': self._identify_behavioral_bias(news_sentiments)
            }
            
        except Exception as e:
            self.logger.error(f"Investor behavior analysis failed: {e}")
            return {
                'fear_greed_index': 50,
                'uncertainty_level': 'medium',
                'herd_behavior': 'moderate',
                'market_mood': 'mixed'
            }
    
    def _calculate_fear_greed_index(self, news_sentiments: List[Dict[str, Any]]) -> int:
        """Fear & Greed Index の計算（0-100）"""
        if not news_sentiments:
            return 50
        
        fear_keywords = self.bias_patterns['fear']
        greed_keywords = self.bias_patterns['greed']
        
        fear_count = 0
        greed_count = 0
        
        for sentiment in news_sentiments:
            text = sentiment['news_item']['cleaned_text']
            
            fear_count += sum(1 for keyword in fear_keywords if keyword in text)
            greed_count += sum(1 for keyword in greed_keywords if keyword in text)
        
        total_emotion = fear_count + greed_count
        if total_emotion == 0:
            return 50  # 中立
        
        greed_ratio = greed_count / total_emotion
        
        # 0-100スケールに変換
        return int(greed_ratio * 100)
    
    def _assess_uncertainty_level(self, news_sentiments: List[Dict[str, Any]]) -> str:
        """不確実性レベルの評価"""
        uncertainty_keywords = self.bias_patterns['uncertainty']
        
        uncertainty_count = 0
        total_news = len(news_sentiments)
        
        for sentiment in news_sentiments:
            text = sentiment['news_item']['cleaned_text']
            uncertainty_count += sum(1 for keyword in uncertainty_keywords if keyword in text)
        
        if total_news == 0:
            return 'medium'
        
        uncertainty_ratio = uncertainty_count / total_news
        
        if uncertainty_ratio > 0.5:
            return 'high'
        elif uncertainty_ratio > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _assess_herd_behavior(self, news_sentiments: List[Dict[str, Any]]) -> str:
        """群集行動の強度評価"""
        if not news_sentiments:
            return 'moderate'
        
        sentiment_scores = [s['weighted_sentiment'] for s in news_sentiments]
        
        # センチメントの分散（群集度の指標）
        if len(sentiment_scores) < 2:
            return 'moderate'
        
        mean_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)
        std_dev = math.sqrt(variance)
        
        # 標準偏差が小さい = 群集行動が強い
        if std_dev < 0.2:
            return 'strong'
        elif std_dev < 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _determine_market_mood(
        self, 
        sentiment_score: float, 
        fear_greed_index: int, 
        uncertainty_level: str
    ) -> str:
        """市場ムードの判定"""
        
        # リスクオン・リスクオフの判定
        if sentiment_score > 0.3 and fear_greed_index > 60 and uncertainty_level == 'low':
            return 'risk_on'
        elif sentiment_score < -0.3 and fear_greed_index < 40 and uncertainty_level == 'high':
            return 'risk_off'
        elif uncertainty_level == 'high' or abs(sentiment_score) < 0.1:
            return 'uncertain'
        else:
            return 'mixed'
    
    def _identify_behavioral_bias(self, news_sentiments: List[Dict[str, Any]]) -> List[str]:
        """行動バイアスの特定"""
        biases = []
        
        if not news_sentiments:
            return biases
        
        # 確認バイアス（同じ方向のニュースばかり）
        positive_count = sum(1 for s in news_sentiments if s['weighted_sentiment'] > 0.2)
        negative_count = sum(1 for s in news_sentiments if s['weighted_sentiment'] < -0.2)
        
        if positive_count > negative_count * 3:
            biases.append('confirmation_bias_bullish')
        elif negative_count > positive_count * 3:
            biases.append('confirmation_bias_bearish')
        
        # リセンシーバイアス（直近ニュースに偏重）
        recent_news = [s for s in news_sentiments if s['news_item']['freshness'] in ['breaking', 'recent']]
        if len(recent_news) > len(news_sentiments) * 0.7:
            biases.append('recency_bias')
        
        # アンカリングバイアス（特定カテゴリに集中）
        categories = [s['news_item']['category'] for s in news_sentiments]
        most_common_category = Counter(categories).most_common(1)[0] if categories else ('', 0)
        if most_common_category[1] > len(categories) * 0.6:
            biases.append('anchoring_bias')
        
        return biases
    
    def _analyze_timing_factors(self, processed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ニュースのタイミング要因分析"""
        try:
            if not processed_news:
                return {
                    'news_freshness': 'unknown',
                    'market_reaction_expected': 'none',
                    'sentiment_duration': 'short'
                }
            
            # 新鮮度の分布
            freshness_counts = Counter(news['freshness'] for news in processed_news)
            most_common_freshness = freshness_counts.most_common(1)[0][0]
            
            # 市場反応の予想
            high_impact_categories = ['central_bank', 'employment', 'inflation']
            has_high_impact = any(
                news['category'] in high_impact_categories for news in processed_news
            )
            
            if most_common_freshness == 'breaking' and has_high_impact:
                market_reaction = 'immediate'
            elif most_common_freshness in ['breaking', 'recent']:
                market_reaction = 'delayed'
            else:
                market_reaction = 'none'
            
            # センチメント持続期間の予想
            if any(news['category'] == 'central_bank' for news in processed_news):
                duration = 'long'
            elif any(news['category'] in ['employment', 'inflation', 'gdp'] for news in processed_news):
                duration = 'medium'
            else:
                duration = 'short'
            
            return {
                'news_freshness': most_common_freshness,
                'market_reaction_expected': market_reaction,
                'sentiment_duration': duration,
                'freshness_distribution': dict(freshness_counts)
            }
            
        except Exception as e:
            self.logger.error(f"Timing analysis failed: {e}")
            return {
                'news_freshness': 'unknown',
                'market_reaction_expected': 'none',
                'sentiment_duration': 'short'
            }
    
    def _detect_contrarian_signals(
        self,
        overall_sentiment: Dict[str, Any],
        investor_behavior: Dict[str, Any],
        market_impact: Dict[str, Any]
    ) -> List[str]:
        """逆張りシグナルの検出"""
        contrarian_signals = []
        
        try:
            sentiment_score = overall_sentiment['score']
            fear_greed_index = investor_behavior.get('fear_greed_index', 50)
            herd_behavior = investor_behavior.get('herd_behavior', 'moderate')
            
            # 極端なセンチメント（逆張り機会）
            if sentiment_score > 0.7 and fear_greed_index > 80:
                contrarian_signals.append('extreme_greed_sell_signal')
            elif sentiment_score < -0.7 and fear_greed_index < 20:
                contrarian_signals.append('extreme_fear_buy_signal')
            
            # 群集行動が強い場合の逆張り
            if herd_behavior == 'strong':
                if sentiment_score > 0.5:
                    contrarian_signals.append('strong_consensus_bullish_fade')
                elif sentiment_score < -0.5:
                    contrarian_signals.append('strong_consensus_bearish_fade')
            
            # 高インパクトだが一方向に偏ったニュース
            impact_level = market_impact.get('level', 1)
            if impact_level >= 4:
                positive_factors = len(overall_sentiment.get('positive_factors', []))
                negative_factors = len(overall_sentiment.get('negative_factors', []))
                
                if positive_factors > negative_factors * 5:
                    contrarian_signals.append('overwhelming_positive_news_fade')
                elif negative_factors > positive_factors * 5:
                    contrarian_signals.append('overwhelming_negative_news_fade')
            
        except Exception as e:
            self.logger.error(f"Contrarian signal detection failed: {e}")
        
        return contrarian_signals
    
    def _generate_sentiment_decision(
        self,
        overall_sentiment: Dict[str, Any],
        sentiment_trend: Dict[str, Any],
        market_impact: Dict[str, Any],
        investor_behavior: Dict[str, Any],
        timing_analysis: Dict[str, Any],
        contrarian_signals: List[str]
    ) -> Dict[str, Any]:
        """センチメントに基づく最終判定"""
        try:
            sentiment_score = overall_sentiment['score']
            trend_direction = sentiment_trend['direction']
            market_mood = investor_behavior['market_mood']
            
            # 基本シグナル
            signals = []
            confidence_factors = []
            key_factors = []
            
            # センチメントスコアからのシグナル
            if sentiment_score > 0.3:
                signals.append('buy')
                confidence_factors.append(min(sentiment_score, 1.0))
                key_factors.append(f'ポジティブセンチメント (スコア: {sentiment_score:.2f})')
            elif sentiment_score < -0.3:
                signals.append('sell')
                confidence_factors.append(min(abs(sentiment_score), 1.0))
                key_factors.append(f'ネガティブセンチメント (スコア: {sentiment_score:.2f})')
            
            # トレンドからのシグナル
            if trend_direction == 'improving':
                signals.append('buy')
                confidence_factors.append(0.6)
                key_factors.append('センチメント改善トレンド')
            elif trend_direction == 'deteriorating':
                signals.append('sell')
                confidence_factors.append(0.6)
                key_factors.append('センチメント悪化トレンド')
            
            # 市場ムードからのシグナル
            if market_mood == 'risk_on':
                signals.append('buy')
                confidence_factors.append(0.5)
                key_factors.append('リスクオンセンチメント')
            elif market_mood == 'risk_off':
                signals.append('sell')
                confidence_factors.append(0.5)
                key_factors.append('リスクオフセンチメント')
            
            # 逆張りシグナルの考慮
            if contrarian_signals:
                for signal in contrarian_signals:
                    if 'sell_signal' in signal or 'fade' in signal:
                        signals.append('sell')
                        confidence_factors.append(0.4)
                        key_factors.append(f'逆張りシグナル: {signal}')
                    elif 'buy_signal' in signal:
                        signals.append('buy')
                        confidence_factors.append(0.4)
                        key_factors.append(f'逆張りシグナル: {signal}')
            
            # 最終判定
            if not signals:
                final_action = 'hold'
                confidence = 0.3
                reasoning = 'センチメント要因に明確な方向性なし'
            else:
                buy_count = signals.count('buy')
                sell_count = signals.count('sell')
                
                if buy_count > sell_count:
                    final_action = 'buy'
                    base_confidence = sum(confidence_factors) / len(confidence_factors)
                    confidence = min(base_confidence + 0.1 * (buy_count - sell_count), 1.0)
                elif sell_count > buy_count:
                    final_action = 'sell'
                    base_confidence = sum(confidence_factors) / len(confidence_factors)
                    confidence = min(base_confidence + 0.1 * (sell_count - buy_count), 1.0)
                else:
                    final_action = 'hold'
                    confidence = sum(confidence_factors) / len(confidence_factors) * 0.8
                
                reasoning = f'{len(key_factors)}個のセンチメント要因に基づく判断'
            
            # 市場インパクトによる信頼度調整
            impact_level = market_impact.get('level', 1)
            if impact_level >= 4:
                confidence *= 1.2  # 高インパクト時は信頼度向上
            elif impact_level <= 2:
                confidence *= 0.8  # 低インパクト時は信頼度低下
            
            confidence = min(confidence, 1.0)
            
            return {
                'action': final_action,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'key_factors': key_factors[:5]
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment decision generation failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f'センチメント判定エラー: {str(e)}',
                'key_factors': []
            }
    
    def _generate_neutral_sentiment_result(self) -> Dict[str, Any]:
        """ニュースデータが無い場合の中立的結果"""
        return {
            'sentiment_score': 0.0,
            'sentiment_trend': 'stable',
            'news_impact_level': 1,
            'market_mood': 'neutral',
            'key_sentiment_drivers': [],
            'news_analysis': {
                'positive_factors': [],
                'negative_factors': [],
                'neutral_factors': []
            },
            'investor_behavior': {
                'fear_greed_index': 50,
                'uncertainty_level': 'medium',
                'herd_behavior': 'moderate',
                'market_mood': 'neutral'
            },
            'timing_analysis': {
                'news_freshness': 'unknown',
                'market_reaction_expected': 'none',
                'sentiment_duration': 'short'
            },
            'action': 'hold',
            'confidence': 0.3,
            'reasoning': 'ニュースデータが不足のため中立判定',
            'key_factors': ['ニュース不足'],
            'contrarian_signals': []
        }