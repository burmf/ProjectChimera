#!/usr/bin/env python3
"""
Fundamental Analysis AI Department (X Department)
ファンダメンタル分析部門AI（X部門）
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os
import re

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_agent_base import AIAgentBase, DepartmentType, AnalysisRequest, AnalysisResult
from core.department_prompts import DepartmentPrompts, PromptFormatter


class FundamentalAnalysisAI(AIAgentBase):
    """
    ファンダメンタル分析専門AIエージェント（X部門）
    
    中央銀行政策、経済指標、地政学的要因の分析を専門とする
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(DepartmentType.FUNDAMENTAL, model_config)
        
        # 経済指標の重要度マッピング
        self.indicator_weights = {
            # 最重要指標（中央銀行政策に直結）
            'interest_rate': 1.0,
            'inflation_rate': 0.9,
            'employment_rate': 0.9,
            'gdp_growth': 0.9,
            
            # 重要指標
            'trade_balance': 0.7,
            'current_account': 0.7,
            'government_debt': 0.6,
            'money_supply': 0.6,
            
            # 参考指標
            'consumer_confidence': 0.4,
            'business_sentiment': 0.4,
            'pmi': 0.5,
            'retail_sales': 0.5
        }
        
        # 通貨ペア別分析の設定
        self.currency_analysis_config = {
            'USD/JPY': {
                'base_currency': 'USD',
                'quote_currency': 'JPY',
                'central_banks': ['Federal Reserve', 'Bank of Japan'],
                'key_factors': [
                    'fed_rate_expectations',
                    'boj_intervention_risk',
                    'us_japan_yield_differential',
                    'risk_sentiment',
                    'carry_trade_flows'
                ]
            },
            'EUR/USD': {
                'base_currency': 'EUR',
                'quote_currency': 'USD',
                'central_banks': ['European Central Bank', 'Federal Reserve'],
                'key_factors': [
                    'ecb_fed_divergence',
                    'eurozone_stability',
                    'dollar_strength_index',
                    'energy_prices'
                ]
            }
        }
        
        # イベントインパクトスコア
        self.event_impact_scores = {
            'central_bank_decision': 0.9,
            'employment_data': 0.8,
            'inflation_data': 0.8,
            'gdp_release': 0.7,
            'geopolitical_event': 0.6,
            'trade_data': 0.5,
            'sentiment_survey': 0.3
        }
        
        self.logger.info("Fundamental Analysis AI (X Department) initialized")
    
    def _get_system_prompt(self) -> str:
        """ファンダメンタル分析専用システムプロンプトを取得"""
        return DepartmentPrompts.get_system_prompt(DepartmentType.FUNDAMENTAL)
    
    def _validate_request_data(self, request: AnalysisRequest) -> bool:
        """リクエストデータの妥当性チェック"""
        try:
            data = request.data
            
            # 最低限のデータ確認
            has_economic_data = 'economic_data' in data
            has_news_data = 'news_data' in data
            has_price_data = 'price_data' in data
            
            if not (has_economic_data or has_news_data or has_price_data):
                self.logger.error("No valid data sources for fundamental analysis")
                return False
            
            # 通貨ペア情報の確認
            symbol = data.get('symbol', 'USD/JPY')
            if symbol not in self.currency_analysis_config:
                self.logger.warning(f"Unsupported currency pair: {symbol}")
                # サポート外でも分析は続行
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
    
    async def _analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """ファンダメンタル分析のメインロジック"""
        try:
            data = request.data
            symbol = data.get('symbol', 'USD/JPY')
            
            # 経済データ分析
            economic_analysis = self._analyze_economic_data(
                data.get('economic_data', {}), symbol
            )
            
            # ニュース分析（ファンダメンタル関連のみ）
            news_analysis = self._analyze_fundamental_news(
                data.get('news_data', []), symbol
            )
            
            # 中央銀行政策分析
            monetary_policy_analysis = self._analyze_monetary_policy(
                economic_analysis, news_analysis, symbol
            )
            
            # 地政学的リスク評価
            geopolitical_analysis = self._analyze_geopolitical_factors(
                news_analysis, symbol
            )
            
            # 通貨相対価値分析
            relative_value_analysis = self._analyze_currency_relative_value(
                economic_analysis, monetary_policy_analysis, symbol
            )
            
            # 時系列インパクト評価
            temporal_impact = self._evaluate_temporal_impact(
                economic_analysis, news_analysis, monetary_policy_analysis
            )
            
            # 総合シグナル生成
            signal = self._generate_fundamental_signal(
                economic_analysis,
                monetary_policy_analysis,
                geopolitical_analysis,
                relative_value_analysis,
                temporal_impact,
                symbol
            )
            
            return {
                'trend': signal['trend'],
                'trend_strength': signal['trend_strength'],
                'drivers': {
                    'monetary_policy': monetary_policy_analysis,
                    'economic_data': economic_analysis,
                    'geopolitical': geopolitical_analysis
                },
                'catalyst_events': temporal_impact['upcoming_events'],
                'time_horizon_impact': temporal_impact['time_horizon_impact'],
                'action': signal['action'],
                'confidence': signal['confidence'],
                'reasoning': signal['reasoning'],
                'key_factors': signal['key_factors'],
                'risk_factors': signal['risk_factors']
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental analysis failed: {e}")
            return {
                'error': str(e),
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f'分析エラー: {str(e)}'
            }
    
    def _analyze_economic_data(
        self, 
        economic_data: Dict[str, Any], 
        symbol: str
    ) -> Dict[str, Any]:
        """経済指標データの分析"""
        try:
            analysis = {
                'recent_surprises': [],
                'outlook': 'stable',
                'strength_score': 0.5,
                'key_indicators': {}
            }
            
            if not economic_data:
                return analysis
            
            surprises = []
            strength_factors = []
            
            # 各経済指標を分析
            for indicator, data in economic_data.items():
                indicator_analysis = self._analyze_single_indicator(
                    indicator, data, symbol
                )
                
                analysis['key_indicators'][indicator] = indicator_analysis
                
                # サプライズ要因の抽出
                if indicator_analysis.get('surprise_factor', 0) > 0.3:
                    surprises.append({
                        'indicator': indicator,
                        'surprise': indicator_analysis['surprise_factor'],
                        'impact': indicator_analysis['market_impact']
                    })
                
                # 通貨強度への寄与
                weight = self.indicator_weights.get(indicator, 0.3)
                contribution = indicator_analysis.get('currency_impact', 0) * weight
                strength_factors.append(contribution)
            
            # 全体的な経済強度スコア
            if strength_factors:
                analysis['strength_score'] = max(0, min(1, 
                    0.5 + sum(strength_factors) / len(strength_factors)
                ))
            
            # 見通し判定
            if analysis['strength_score'] > 0.6:
                analysis['outlook'] = 'improving'
            elif analysis['strength_score'] < 0.4:
                analysis['outlook'] = 'deteriorating'
            
            analysis['recent_surprises'] = sorted(
                surprises, key=lambda x: x['surprise'], reverse=True
            )[:5]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Economic data analysis failed: {e}")
            return {'outlook': 'stable', 'strength_score': 0.5, 'recent_surprises': []}
    
    def _analyze_single_indicator(
        self, 
        indicator: str, 
        data: Any, 
        symbol: str
    ) -> Dict[str, Any]:
        """単一経済指標の分析"""
        try:
            analysis = {
                'value': data if isinstance(data, (int, float)) else 0,
                'trend': 'neutral',
                'surprise_factor': 0.0,
                'market_impact': 'neutral',
                'currency_impact': 0.0
            }
            
            # データが辞書形式の場合（実際値、予想値、前回値含む）
            if isinstance(data, dict):
                actual = data.get('actual', 0)
                forecast = data.get('forecast', actual)
                previous = data.get('previous', actual)
                
                analysis['value'] = actual
                
                # サプライズファクター計算
                if forecast != 0:
                    analysis['surprise_factor'] = abs(actual - forecast) / abs(forecast)
                
                # トレンド判定
                if actual > previous:
                    analysis['trend'] = 'improving'
                    analysis['currency_impact'] = 0.3
                elif actual < previous:
                    analysis['trend'] = 'deteriorating'  
                    analysis['currency_impact'] = -0.3
                
                # 市場インパクト評価
                if analysis['surprise_factor'] > 0.5:
                    analysis['market_impact'] = 'high'
                elif analysis['surprise_factor'] > 0.2:
                    analysis['market_impact'] = 'medium'
                else:
                    analysis['market_impact'] = 'low'
            
            # 指標固有の分析ロジック
            analysis.update(self._apply_indicator_specific_logic(indicator, analysis))
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Single indicator analysis failed for {indicator}: {e}")
            return {'trend': 'neutral', 'surprise_factor': 0.0, 'currency_impact': 0.0}
    
    def _apply_indicator_specific_logic(
        self, 
        indicator: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """指標固有のロジックを適用"""
        updates = {}
        
        try:
            value = analysis['value']
            
            if 'inflation' in indicator.lower():
                # インフレ率の分析
                if value > 3.0:  # 高インフレ
                    updates['interpretation'] = 'high_inflation_risk'
                    updates['currency_impact'] = 0.4  # 金利上昇期待
                elif value < 1.0:  # 低インフレ
                    updates['interpretation'] = 'low_inflation_risk'
                    updates['currency_impact'] = -0.3  # 金利据え置き期待
                
            elif 'employment' in indicator.lower() or 'unemployment' in indicator.lower():
                # 雇用関連指標
                if 'unemployment' in indicator.lower():
                    # 失業率（低い方が良い）
                    if value < 4.0:
                        updates['interpretation'] = 'strong_labor_market'
                        updates['currency_impact'] = 0.3
                    elif value > 7.0:
                        updates['interpretation'] = 'weak_labor_market'
                        updates['currency_impact'] = -0.4
                else:
                    # 雇用者数（高い方が良い）
                    if value > 200000:  # 月次雇用者数（米国）
                        updates['interpretation'] = 'strong_job_growth'
                        updates['currency_impact'] = 0.3
                
            elif 'gdp' in indicator.lower():
                # GDP成長率
                if value > 3.0:
                    updates['interpretation'] = 'strong_growth'
                    updates['currency_impact'] = 0.4
                elif value < 1.0:
                    updates['interpretation'] = 'weak_growth'
                    updates['currency_impact'] = -0.3
                
            elif 'trade_balance' in indicator.lower():
                # 貿易収支
                if value > 0:
                    updates['interpretation'] = 'trade_surplus'
                    updates['currency_impact'] = 0.2
                else:
                    updates['interpretation'] = 'trade_deficit'
                    updates['currency_impact'] = -0.2
                    
        except Exception as e:
            self.logger.error(f"Indicator-specific logic failed for {indicator}: {e}")
        
        return updates
    
    def _analyze_fundamental_news(
        self, 
        news_data: List[Dict[str, Any]], 
        symbol: str
    ) -> Dict[str, Any]:
        """ファンダメンタル関連ニュースの分析"""
        try:
            analysis = {
                'total_news_count': len(news_data),
                'fundamental_news_count': 0,
                'sentiment_score': 0.0,
                'key_themes': [],
                'central_bank_news': [],
                'economic_surprises': [],
                'policy_implications': []
            }
            
            if not news_data:
                return analysis
            
            fundamental_news = []
            sentiment_scores = []
            themes = {}
            
            for news in news_data:
                if self._is_fundamental_news(news):
                    fundamental_news.append(news)
                    
                    # ニュースのカテゴリ分析
                    category = self._categorize_fundamental_news(news)
                    
                    # テーマ集計
                    theme = self._extract_news_theme(news)
                    if theme:
                        themes[theme] = themes.get(theme, 0) + 1
                    
                    # センチメントスコア
                    sentiment = self._calculate_news_sentiment(news, symbol)
                    sentiment_scores.append(sentiment)
                    
                    # 中央銀行関連ニュースの特別処理
                    if self._is_central_bank_news(news):
                        analysis['central_bank_news'].append({
                            'title': news.get('title', ''),
                            'sentiment': sentiment,
                            'category': category,
                            'timestamp': news.get('published_at', '')
                        })
            
            analysis['fundamental_news_count'] = len(fundamental_news)
            
            # 全体的なセンチメント
            if sentiment_scores:
                analysis['sentiment_score'] = sum(sentiment_scores) / len(sentiment_scores)
            
            # 主要テーマ（上位3つ）
            analysis['key_themes'] = sorted(
                themes.items(), key=lambda x: x[1], reverse=True
            )[:3]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fundamental news analysis failed: {e}")
            return {'total_news_count': 0, 'sentiment_score': 0.0}
    
    def _is_fundamental_news(self, news: Dict[str, Any]) -> bool:
        """ニュースがファンダメンタル分析に関連するかチェック"""
        fundamental_keywords = [
            # 中央銀行関連
            'federal reserve', 'fed', 'central bank', 'interest rate', 'monetary policy',
            'bank of japan', 'boj', 'ecb', 'european central bank',
            
            # 経済指標
            'gdp', 'inflation', 'cpi', 'ppi', 'employment', 'unemployment', 'payroll',
            'retail sales', 'consumer confidence', 'business sentiment', 'pmi',
            'trade balance', 'current account', 'budget', 'debt',
            
            # 政策・政治
            'fiscal policy', 'government', 'election', 'policy', 'regulation',
            'trade war', 'tariff', 'sanctions',
            
            # 経済全般
            'economic growth', 'recession', 'recovery', 'outlook', 'forecast'
        ]
        
        content = (news.get('title', '') + ' ' + news.get('content', '')).lower()
        
        return any(keyword in content for keyword in fundamental_keywords)
    
    def _categorize_fundamental_news(self, news: Dict[str, Any]) -> str:
        """ファンダメンタルニュースのカテゴリ分類"""
        content = (news.get('title', '') + ' ' + news.get('content', '')).lower()
        
        if any(keyword in content for keyword in ['federal reserve', 'fed', 'central bank', 'interest rate']):
            return 'monetary_policy'
        elif any(keyword in content for keyword in ['gdp', 'growth', 'recession']):
            return 'economic_growth'
        elif any(keyword in content for keyword in ['inflation', 'cpi', 'ppi']):
            return 'inflation'
        elif any(keyword in content for keyword in ['employment', 'unemployment', 'job']):
            return 'employment'
        elif any(keyword in content for keyword in ['trade', 'tariff', 'export', 'import']):
            return 'trade'
        elif any(keyword in content for keyword in ['government', 'policy', 'election']):
            return 'fiscal_policy'
        else:
            return 'other'
    
    def _extract_news_theme(self, news: Dict[str, Any]) -> Optional[str]:
        """ニュースからテーマを抽出"""
        content = (news.get('title', '') + ' ' + news.get('content', '')).lower()
        
        theme_patterns = {
            'rate_hike_expectations': ['rate hike', 'raise rates', 'tighten policy'],
            'inflation_concerns': ['inflation', 'price pressure', 'cost of living'],
            'economic_slowdown': ['slowdown', 'recession', 'contraction'],
            'trade_tensions': ['trade war', 'tariff', 'trade dispute'],
            'geopolitical_risk': ['geopolitical', 'conflict', 'tension', 'war'],
            'energy_crisis': ['energy', 'oil price', 'gas price'],
            'dollar_strength': ['dollar strength', 'usd strength', 'strong dollar']
        }
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in content for keyword in keywords):
                return theme
        
        return None
    
    def _calculate_news_sentiment(self, news: Dict[str, Any], symbol: str) -> float:
        """ニュースのセンチメントスコア計算"""
        content = (news.get('title', '') + ' ' + news.get('content', '')).lower()
        
        positive_keywords = [
            'growth', 'increase', 'rise', 'improve', 'strong', 'positive',
            'recovery', 'expansion', 'boost', 'gain', 'surge', 'rally'
        ]
        
        negative_keywords = [
            'decline', 'fall', 'drop', 'weak', 'negative', 'recession',
            'slowdown', 'concern', 'risk', 'uncertainty', 'crisis', 'collapse'
        ]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in content)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content)
        
        # -1.0 から 1.0 のスコア
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
    def _is_central_bank_news(self, news: Dict[str, Any]) -> bool:
        """中央銀行関連ニュースかチェック"""
        content = (news.get('title', '') + ' ' + news.get('content', '')).lower()
        
        cb_keywords = [
            'federal reserve', 'fed', 'jerome powell', 'fomc',
            'bank of japan', 'boj', 'kazuo ueda',
            'ecb', 'european central bank', 'christine lagarde',
            'bank of england', 'boe',
            'people\'s bank of china', 'pboc'
        ]
        
        return any(keyword in content for keyword in cb_keywords)
    
    def _analyze_monetary_policy(
        self,
        economic_analysis: Dict[str, Any],
        news_analysis: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """金融政策分析"""
        try:
            config = self.currency_analysis_config.get(symbol, {})
            base_currency = config.get('base_currency', 'USD')
            quote_currency = config.get('quote_currency', 'JPY')
            
            analysis = {
                'base_currency': 'neutral',
                'quote_currency': 'neutral',
                'differential_impact': 'neutral',
                'rate_expectations': {},
                'policy_divergence_score': 0.0
            }
            
            # 経済データから政策スタンスを推定
            base_stance = self._estimate_policy_stance(economic_analysis, base_currency)
            quote_stance = self._estimate_policy_stance(economic_analysis, quote_currency)
            
            analysis['base_currency'] = base_stance
            analysis['quote_currency'] = quote_stance
            
            # 政策差による影響
            stance_scores = {'tightening': 1, 'neutral': 0, 'easing': -1}
            base_score = stance_scores.get(base_stance, 0)
            quote_score = stance_scores.get(quote_stance, 0)
            
            differential = base_score - quote_score
            if differential > 0:
                analysis['differential_impact'] = 'positive'  # 基軸通貨に有利
            elif differential < 0:
                analysis['differential_impact'] = 'negative'  # 基軸通貨に不利
            
            analysis['policy_divergence_score'] = differential / 2.0  # -1.0 to 1.0
            
            # 中央銀行ニュースからの追加情報
            cb_news = news_analysis.get('central_bank_news', [])
            if cb_news:
                cb_sentiment = sum(news['sentiment'] for news in cb_news) / len(cb_news)
                analysis['cb_news_sentiment'] = cb_sentiment
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Monetary policy analysis failed: {e}")
            return {'differential_impact': 'neutral', 'policy_divergence_score': 0.0}
    
    def _estimate_policy_stance(self, economic_analysis: Dict[str, Any], currency: str) -> str:
        """経済データから政策スタンスを推定"""
        try:
            key_indicators = economic_analysis.get('key_indicators', {})
            
            tightening_signals = 0
            easing_signals = 0
            
            # インフレ率チェック
            for indicator_name, data in key_indicators.items():
                if 'inflation' in indicator_name.lower():
                    value = data.get('value', 0)
                    if value > 3.0:  # 高インフレ→引き締め
                        tightening_signals += 1
                    elif value < 1.0:  # 低インフレ→緩和
                        easing_signals += 1
                
                # 雇用データ
                elif 'employment' in indicator_name.lower():
                    trend = data.get('trend', 'neutral')
                    if trend == 'improving':
                        tightening_signals += 0.5
                    elif trend == 'deteriorating':
                        easing_signals += 0.5
                
                # GDP成長率
                elif 'gdp' in indicator_name.lower():
                    value = data.get('value', 0)
                    if value > 3.0:  # 高成長→引き締め
                        tightening_signals += 0.5
                    elif value < 1.0:  # 低成長→緩和
                        easing_signals += 0.5
            
            # 政策スタンス決定
            if tightening_signals > easing_signals + 0.5:
                return 'tightening'
            elif easing_signals > tightening_signals + 0.5:
                return 'easing'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _analyze_geopolitical_factors(
        self,
        news_analysis: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """地政学的要因の分析"""
        try:
            analysis = {
                'risk_level': 'low',
                'key_factors': [],
                'safe_haven_impact': 'neutral'
            }
            
            key_themes = news_analysis.get('key_themes', [])
            risk_factors = []
            
            # リスクテーマの検出
            high_risk_themes = ['trade_tensions', 'geopolitical_risk', 'energy_crisis']
            medium_risk_themes = ['election_uncertainty', 'policy_change']
            
            for theme, count in key_themes:
                if theme in high_risk_themes:
                    risk_factors.append({'theme': theme, 'severity': 'high', 'frequency': count})
                elif theme in medium_risk_themes:
                    risk_factors.append({'theme': theme, 'severity': 'medium', 'frequency': count})
            
            # リスクレベルの決定
            if any(factor['severity'] == 'high' for factor in risk_factors):
                analysis['risk_level'] = 'high'
            elif risk_factors:
                analysis['risk_level'] = 'medium'
            
            analysis['key_factors'] = [factor['theme'] for factor in risk_factors]
            
            # セーフヘイブン効果の分析
            if symbol in ['USD/JPY', 'EUR/USD']:
                if analysis['risk_level'] == 'high':
                    # 通常、リスクオフ時はUSD、JPYが強くなる
                    if 'JPY' in symbol:
                        analysis['safe_haven_impact'] = 'jpy_positive'
                    else:
                        analysis['safe_haven_impact'] = 'usd_positive'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Geopolitical analysis failed: {e}")
            return {'risk_level': 'low', 'key_factors': []}
    
    def _analyze_currency_relative_value(
        self,
        economic_analysis: Dict[str, Any],
        monetary_policy_analysis: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """通貨の相対価値分析"""
        try:
            config = self.currency_analysis_config.get(symbol, {})
            base_currency = config.get('base_currency', 'USD')
            quote_currency = config.get('quote_currency', 'JPY')
            
            analysis = {
                'relative_strength': 'neutral',
                'economic_score_differential': 0.0,
                'policy_score_differential': 0.0,
                'overall_bias': 'neutral'
            }
            
            # 経済強度差
            base_strength = economic_analysis.get('strength_score', 0.5)
            # quote通貨のスコアは逆数で計算（簡略化）
            quote_strength = 1.0 - base_strength
            
            economic_differential = base_strength - quote_strength
            analysis['economic_score_differential'] = economic_differential
            
            # 政策スコア差
            policy_differential = monetary_policy_analysis.get('policy_divergence_score', 0.0)
            analysis['policy_score_differential'] = policy_differential
            
            # 総合判定
            total_score = (economic_differential + policy_differential) / 2.0
            
            if total_score > 0.2:
                analysis['relative_strength'] = f'{base_currency}_strong'
                analysis['overall_bias'] = 'bullish'
            elif total_score < -0.2:
                analysis['relative_strength'] = f'{quote_currency}_strong' 
                analysis['overall_bias'] = 'bearish'
            else:
                analysis['relative_strength'] = 'balanced'
                analysis['overall_bias'] = 'neutral'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Relative value analysis failed: {e}")
            return {'relative_strength': 'neutral', 'overall_bias': 'neutral'}
    
    def _evaluate_temporal_impact(
        self,
        economic_analysis: Dict[str, Any],
        news_analysis: Dict[str, Any],
        monetary_policy_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """時系列インパクト評価"""
        try:
            analysis = {
                'upcoming_events': [],
                'time_horizon_impact': {
                    '1_week': 'neutral',
                    '1_month': 'neutral',
                    '3_months': 'neutral'
                }
            }
            
            # 今後のイベント予想（簡略化）
            upcoming_events = []
            
            # 中央銀行イベント
            cb_news_count = len(news_analysis.get('central_bank_news', []))
            if cb_news_count > 0:
                upcoming_events.append('central_bank_meeting')
            
            # 経済指標発表
            key_indicators = economic_analysis.get('key_indicators', {})
            if any('employment' in name.lower() for name in key_indicators.keys()):
                upcoming_events.append('employment_data_release')
            
            if any('inflation' in name.lower() for name in key_indicators.keys()):
                upcoming_events.append('inflation_data_release')
            
            analysis['upcoming_events'] = upcoming_events
            
            # 時間軸別インパクト
            policy_divergence = monetary_policy_analysis.get('policy_divergence_score', 0.0)
            
            # 短期（1週間）：ニュースとイベント主導
            if abs(policy_divergence) > 0.3:
                analysis['time_horizon_impact']['1_week'] = 'positive' if policy_divergence > 0 else 'negative'
            
            # 中期（1ヶ月）：経済データとトレンド
            economic_outlook = economic_analysis.get('outlook', 'stable')
            if economic_outlook == 'improving':
                analysis['time_horizon_impact']['1_month'] = 'positive'
            elif economic_outlook == 'deteriorating':
                analysis['time_horizon_impact']['1_month'] = 'negative'
            
            # 長期（3ヶ月）：構造的要因
            if abs(policy_divergence) > 0.5:
                analysis['time_horizon_impact']['3_months'] = 'positive' if policy_divergence > 0 else 'negative'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Temporal impact evaluation failed: {e}")
            return {'upcoming_events': [], 'time_horizon_impact': {}}
    
    def _generate_fundamental_signal(
        self,
        economic_analysis: Dict[str, Any],
        monetary_policy_analysis: Dict[str, Any],
        geopolitical_analysis: Dict[str, Any],
        relative_value_analysis: Dict[str, Any],
        temporal_impact: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """ファンダメンタル総合シグナル生成"""
        try:
            # シグナル要素の収集
            signals = []
            confidence_factors = []
            key_factors = []
            risk_factors = []
            
            # 経済データからのシグナル
            economic_outlook = economic_analysis.get('outlook', 'stable')
            if economic_outlook == 'improving':
                signals.append('buy')
                confidence_factors.append(0.6)
                key_factors.append('経済データ改善傾向')
            elif economic_outlook == 'deteriorating':
                signals.append('sell')
                confidence_factors.append(0.6)
                key_factors.append('経済データ悪化傾向')
            
            # 金融政策からのシグナル
            policy_differential = monetary_policy_analysis.get('policy_divergence_score', 0.0)
            if policy_differential > 0.3:
                signals.append('buy')
                confidence_factors.append(0.8)
                key_factors.append('金融政策の相対的引き締め')
            elif policy_differential < -0.3:
                signals.append('sell')
                confidence_factors.append(0.8)
                key_factors.append('金融政策の相対的緩和')
            
            # 相対価値からのシグナル
            overall_bias = relative_value_analysis.get('overall_bias', 'neutral')
            if overall_bias == 'bullish':
                signals.append('buy')
                confidence_factors.append(0.7)
                key_factors.append('通貨相対価値で優位')
            elif overall_bias == 'bearish':
                signals.append('sell')
                confidence_factors.append(0.7)
                key_factors.append('通貨相対価値で劣位')
            
            # 地政学的リスクの影響
            geo_risk_level = geopolitical_analysis.get('risk_level', 'low')
            if geo_risk_level == 'high':
                risk_factors.append('高い地政学リスク')
                # リスクオフ時のセーフヘイブン効果
                safe_haven_impact = geopolitical_analysis.get('safe_haven_impact', 'neutral')
                if safe_haven_impact != 'neutral':
                    if 'positive' in safe_haven_impact:
                        signals.append('buy')
                        confidence_factors.append(0.5)
                        key_factors.append('セーフヘイブン需要')
            
            # 最終シグナル決定
            if not signals:
                final_action = 'hold'
                confidence = 0.4
                reasoning = 'ファンダメンタル要因に明確な方向性なし'
                trend = 'neutral'
                trend_strength = 0.5
            else:
                buy_count = signals.count('buy')
                sell_count = signals.count('sell')
                
                if buy_count > sell_count:
                    final_action = 'buy'
                    trend = 'bullish'
                    base_confidence = sum(confidence_factors) / len(confidence_factors)
                    confidence = min(base_confidence + 0.1 * (buy_count - sell_count), 1.0)
                elif sell_count > buy_count:
                    final_action = 'sell'
                    trend = 'bearish'
                    base_confidence = sum(confidence_factors) / len(confidence_factors)
                    confidence = min(base_confidence + 0.1 * (sell_count - buy_count), 1.0)
                else:
                    final_action = 'hold'
                    trend = 'neutral'
                    confidence = sum(confidence_factors) / len(confidence_factors) * 0.8
                
                reasoning = f'{len(key_factors)}個のファンダメンタル要因に基づく総合判断'
                
                # トレンド強度
                trend_strength = min(confidence, 1.0)
            
            # リスクファクターによる信頼度調整
            if risk_factors:
                confidence *= 0.8  # リスクがある場合は信頼度を下げる
            
            return {
                'action': final_action,
                'trend': trend,
                'trend_strength': float(trend_strength),
                'confidence': float(confidence),
                'reasoning': reasoning,
                'key_factors': key_factors[:5],
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental signal generation failed: {e}")
            return {
                'action': 'hold',
                'trend': 'neutral',
                'trend_strength': 0.5,
                'confidence': 0.0,
                'reasoning': f'シグナル生成エラー: {str(e)}',
                'key_factors': [],
                'risk_factors': []
            }