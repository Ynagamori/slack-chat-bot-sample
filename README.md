## 概要

このリポジトリは、Slack からの質問に対して社内ドキュメント（`docs` 配下）を検索し、
Ollama + pgvector を使って回答する QA ボットです。

- LLM 推論: Ollama（`/api/generate`）
- 埋め込み: Ollama の embeddings API
- ベクトル DB: PostgreSQL + pgvector
- インターフェース: Slack Bot（Socket Mode）

AIエージェントの導入もスムーズにするため、各種mdも追加しています。

## 環境構築

### 1. 前提

- Docker / Docker Compose がインストールされていること
- Slack App（Bot）を作成済みで、以下のトークンを取得済みであること
  - `SLACK_BOT_TOKEN`
  - `SLACK_APP_TOKEN`


### 2. `.env` の作成

`.env.example` をベースに `.env` を作成します。

```bash
cp .env.example .env
```

`.env` を開いて、Slack と DB / モデルの設定を行います。

```env
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...

DB_HOST=pgvector-db
DB_USER=llm
DB_PASSWORD=llm_pass
DB_NAME=llmdb

LLM_MODEL=llama3.2:3b-instruct-q4_0  # デフォルト値。Ollama に pull 済みのモデル名
EMBED_MODEL=nomic-embed-text        # Ollama に pull 済みの埋め込みモデル名
```

※ `LLM_MODEL` / `EMBED_MODEL` は、実際に `ollama list` に出てくるモデル名に合わせてください。


### 3. コンテナの起動

```bash
docker compose up -d --build
```

初回起動時は、Ollama コンテナ側で使用するモデルを pull しておきます。

```bash
docker compose exec ollama ollama pull llama3.2:3b-instruct-q4_0
docker compose exec ollama ollama pull nomic-embed-text
```

モデルの一覧確認:

```bash
docker compose exec ollama ollama list
```


### 4. docs の取り込み（ベクトル化）

`docs` ディレクトリ配下の `.md` / `.mdx` / `.txt` をチャンク（一定文字数ごとの断片）に分割し、
各チャンクを埋め込みベクトルに変換して DB に登録します。

```bash
docker compose exec app python ingest_docs.py
```

`ingest_docs.py` は、ファイル内容のハッシュを使って変更有無を判定し、  
**新規または内容が更新されたファイルのみ** を再インデックスします（変更のないファイルはスキップされます）。  
また、`docs/` から削除されたファイルに対応するレコードは、自動的に `documents` テーブルからも削除されます。


### 5. Slack Bot の動作

コンテナ起動後、Slack で Bot を招待したチャンネルでメンションすると、
関連ドキュメントを検索しつつ LLM に問い合わせて回答を返します。

```text
@your-bot-name 社内 VPN の接続手順を教えて
```


## RAG の動きと「学習」のさせ方

### 検索〜回答までの流れ

- ユーザーの質問文をそのまま検索クエリとして利用
- `search_similar_docs` で、まず全文を対象にしたキーワード検索を行い、それでも足りない場合にベクトル検索で補完
- 上位数件（デフォルト3件）のチャンクをコンテキストとして `build_prompt` に渡し、Ollama に質問
- LLM 呼び出し時にタイムアウトやエラーが発生した場合は、スタックトレースではなく日本語のエラーメッセージを Slack に返します

### FAQ で精度を高める（擬似的な学習）

1. 精度を高めたい質問を FAQ として `docs/99_faq/` に追加する  
   例: `docs/99_faq/01_wifi_faq.txt`

   ```text
   Q1. 社内Wi-Fiのパスワードを教えて。
   A1. 社内ネットワーク（従業員用）のSSIDは「Corp_Secure_v2」、パスワードは「!S3cure#Pass2025」です。ゲスト用ネットワーク（SSID: Corp_Guest / パスワード: welcome-guest）と取り違えないように注意してください。
   ```

2. 再インデックスを実行

   ```bash
   docker compose exec app python ingest_docs.py
   ```

3. Slack から同じ質問を投げて、期待通りの回答になっているか確認する

誤回答が出た場合は、正しい回答の FAQ を 1〜2 問ずつ増やしていく運用で、段階的に精度を上げていくことができます。


### タイムアウトが気になる場合のヒント

- `.env` の `LLM_MODEL` を軽めのモデルにする  
  （例: 大きい `qwen2.5:7b` ではなく、`llama3.2:3b` など）
- コンテキスト量を増やしすぎない  
  （すでに上位3件に絞るようにしてありますが、さらに長文ドキュメントを細かく分割するのも有効です）
- それでも 120 秒タイムアウトが頻発する場合は、`app/main.py` の `call_llm` で `num_predict` をさらに小さくするか、`timeout` を延ばすこともできます。


## ディレクトリ構成と説明

ルートディレクトリの構成は次の通りです。

```text
.
├── README.md           # このファイル
├── docker-compose.yml  # 各コンテナ定義（DB / Ollama / app）
├── .env                # 実行時の環境変数（Git 管理外を想定）
├── .env.example        # .env のサンプル
├── app/                # アプリケーション本体（Slack Bot + ingest スクリプト）
├── docs/               # 取り込み対象の社内ドキュメント
├── sql/                # DB 初期化スクリプト（pgvector など）
└── data/               # 永続化データ（Postgres, Ollama モデル）
```

### `app/` ディレクトリ

Slack Bot 本体と、ドキュメント取り込みスクリプトが入っています。

```text
app/
├── Dockerfile        # app サービス用 Dockerfile
├── requirements.txt  # Python 依存パッケージ
├── main.py           # Slack Bot + LLM 呼び出し処理
└── ingest_docs.py    # docs を pgvector に登録するスクリプト
```

- `main.py`
  - 起動時に Ollama API が応答するまで待機
  - PostgreSQL（pgvector）に接続
  - Slack Bolt を使って `app_mention` イベントを受け取り
  - ユーザー質問をそのまま検索クエリとして利用し、キーワード検索＋ベクトル検索で関連ドキュメントを取得
  - 取得したチャンク群をコンテキストにして LLM へ投げて回答を Slack に投稿（タイムアウト時は日本語のエラーメッセージを返す）

- `ingest_docs.py`
  - `docs/` 以下の `.txt`, `.md`, `.mdx` を再帰的に走査
  - 各ファイルを一定文字数ごとのチャンクに分割
  - 各チャンクを Ollama embeddings API でベクトル化
  - `documents` テーブルに `title`, `content`（チャンク単位）, `embedding`, `metadata`（path, chunk 番号）を保存
  - FAQ などを追加したあとに再実行すると、新しい知識を「追学習」的に RAG に反映できます


### `docs/` ディレクトリ

- 社内ドキュメントを配置するディレクトリです。
- Markdown (`.md`, `.mdx`) やテキスト (`.txt`) ファイルが対象になります。
- 階層構造は自由に追加して構いません（再帰的に取り込みます）。
- 精度を高めたい Q&A は `docs/99_faq/` 配下に FAQ ファイルとして追加しておくと、RAG の検索対象として利用されます。

例:

```text
docs/
├── README.md
├── infra/
│   └── vpn.md
├── rules/
│   └── security.md
└── 99_faq/

```


### `sql/` ディレクトリ

```text
sql/
└── init.sql  # pgvector 拡張と documents テーブル定義
```

- コンテナ起動時に `pgvector-db` にマウントされ、初期化 SQL として実行されます。
- `documents` テーブル（768 次元の `VECTOR` カラム）と ivfflat インデックスを作成します。


### `data/` ディレクトリ

```text
data/
├── postgres/  # PostgreSQL データ永続化用（コンテナからマウントされる）
└── ollama/    # Ollama モデル・設定の永続化
    ├── models/
    ├── id_ed25519
    └── id_ed25519.pub
```

- `docker-compose.yml` の `volumes` で各コンテナにマウントされます。
- 実運用では、このディレクトリは Git 管理外にしておくことを推奨します。


## よく使うコマンド

- コンテナの起動 / 停止

  ```bash
  docker compose up -d --build   # 起動
  docker compose down            # 停止
  ```

- ログ確認

  ```bash
  docker compose logs -f app
  docker compose logs -f ollama
  docker compose logs -f pgvector-db
  ```

- DB 内の `documents` 件数確認（psql から）

  ```bash
  docker compose exec pgvector-db psql -U llm -d llmdb -c "SELECT COUNT(*) FROM documents;"
  ```

