# SKILLS.md — sql ディレクトリ

データベース初期化スクリプト（pgvector 拡張・`documents` テーブル定義）を配置するディレクトリです。  
PostgreSQL + pgvector を使った RAG の「土台」となる部分です。

## ファイル別の役割

### `init.sql`

- 拡張機能
  - `CREATE EXTENSION IF NOT EXISTS vector;` により pgvector 拡張を有効化。
- テーブル定義
  - `documents` テーブル
    - `id SERIAL PRIMARY KEY`
    - `title TEXT`
    - `content TEXT`（チャンク単位の本文）
    - `embedding VECTOR(768)`（埋め込みベクトル）
    - `metadata JSONB`（`path` / `chunk` など）
- インデックス
  - `documents_embedding_idx` によって `embedding` に ivfflat インデックスを作成。
  - `vector_l2_ops` / `WITH (lists = 100)` で典型的な近傍検索用の設定。

## このディレクトリで得意なこと

- アプリコードとの整合性チェック
  - `app/ingest_docs.py` が `embedding VECTOR(768)` 前提で動いているか確認する。
  - `metadata` の構造（`path` / `chunk`）が `ingest_docs.py` と一致しているか確認する。
  - `search_similar_docs` の SQL（`ORDER BY embedding <-> %s`）とインデックスが噛み合っているか確認する。
- embeddings モデル変更の影響評価
  - 埋め込みモデルを変更したい場合に、必要なベクトル次元数（`VECTOR(N)`）を算出し、  
    `init.sql` / `ingest_docs.py` / `.env` / `README.md` のどこを更新すべきかを洗い出す。

## 作業フローの例

- 「埋め込みモデルを変えたい」場合
  1. 新しいモデルのベクトル次元数を調査する（Ollama のドキュメントなど）。
  2. `init.sql` の `VECTOR(768)` を新しい次元数に変更する。
  3. `.env` の `EMBED_MODEL` を新モデルに変更する。
  4. 既存 DB を作り直す（`docker compose down -v` など）か、`documents` テーブルを再作成する。
  5. `docker compose up -d --build` → `ingest_docs.py` を実行して再インデックス。
  6. `README.md` に変更内容を反映する。

## 作業上の注意

- 既存データベースに影響するスキーマ変更（カラム追加・型変更など）は慎重に行う。
- `VECTOR(768)` のようなモデル依存パラメータを変える場合は、アプリ側の埋め込みモデル・再インデックス手順も併せて更新する。
- 本番環境がある場合は、マイグレーション戦略（別テーブルへの移行・バックフィル）を検討し、`init.sql` だけで解決しようとしない。
