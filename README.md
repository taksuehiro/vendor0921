# AIベンダー検索 (Bedrock版)

## 概要
Markdownファイルに記録したベンダー情報を読み込み、AWS Bedrockを利用して検索・要約するStreamlitアプリです。

## 必要環境
- Python 3.9+
- AWS CLI設定済み (`aws configure`でアクセスキーを登録)
- IAMでBedrock利用権限を持っていること

## セットアップ
```bash
git clone https://github.com/taksuehiro/vendor0921.git
cd vendor0921
pip install -r requirements.txt
```

## 実行
```bash
streamlit run app_bedrock.py --server.port 8080 --server.address 0.0.0.0
```

## 機能
- ベンダー情報の自然言語検索
- AWS Bedrock Claude-v2による高精度な回答生成
- Amazon Titan Embeddingsによる高速ベクトル検索
- FAISSベクトルストアによる永続化
- ベンダーカード形式での結果表示
- 参照ソースの抜粋表示

## 使用モデル
- **LLM**: anthropic.claude-v2
- **Embeddings**: amazon.titan-embed-text-v1
- **リージョン**: us-east-1

## ファイル構成
- `app_bedrock.py` - メインアプリケーション
- `requirements.txt` - 必要ライブラリ
- `data/ベンダー調査.md` - ベンダー情報データ
- `vectorstore/` - FAISSベクトルストア（自動生成）

## 注意事項
- 初回起動時にベクトルストアが作成されます
- AWS認証情報が正しく設定されている必要があります
- Bedrockの利用料金が発生します
