import pytest
from unittest.mock import patch, Mock
from fetch_news import fetch_news
import os
import pandas as pd

@patch('requests.get')
@patch('pandas.DataFrame.to_csv')
def test_fetch_news_success(mock_to_csv, mock_get):
    # 正常系テスト
    mock_response = Mock()
    mock_response.json.return_value = {
        "articles": [{
            "publishedAt": "2025-05-18T03:21:51Z",
            "title": "テスト記事",
            "content": "為替相場のテストコンテンツ",
            "source": {"name": "テストニュース"},
            "url": "http://example.com"
        }]
    }
    mock_get.return_value = mock_response

    fetch_news()
    
    # 保存処理が呼ばれたことを確認
    mock_to_csv.assert_called_once()
    assert "data/raw/news_20250518.csv" in mock_to_csv.call_args[0][0]

@patch('requests.get')
def test_fetch_news_api_error(mock_get):
    # APIエラーテスト
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("API接続エラー")
    
    with patch('builtins.print') as mock_print:
        fetch_news()
        mock_print.assert_called()

@patch('requests.get')
def test_fetch_news_empty_response(mock_get):
    # 空レスポンステスト
    mock_response = Mock()
    mock_response.json.return_value = {"articles": []}
    mock_get.return_value = mock_response

    with patch('builtins.print') as mock_print:
        fetch_news()
        mock_print.assert_called_with("取得完了: 0件のニュースを保存")

@patch('requests.get')
def test_fetch_news_retry_success(mock_get):
    # リトライ成功テスト（2回失敗後成功）
    import requests
    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Timeout"),
        requests.exceptions.ConnectionError("Connection Error"),
        Mock(json=Mock(return_value={"articles": []}))
    ]
    
    with patch('tenacity.nap.time') as mock_sleep:
        fetch_news()
        assert mock_get.call_count == 3
        mock_sleep.assert_called()

@patch('requests.get')
def test_fetch_news_retry_failure(mock_get):
    # リトライ失敗テスト
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("API Error")
    
    with patch('builtins.print') as mock_print:
        fetch_news()
        assert mock_get.call_count == 3
        mock_print.assert_called()

def test_env_variable_loaded():
    # 環境変数読み込みテスト
    assert 'NEWS_API_KEY' in os.environ