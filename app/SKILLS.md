# SKILLS.md — app ディレクトリ

このディレクトリには、Slack Bot 本体とドキュメント取り込みスクリプトが含まれます。  
RAG ロジックや会話制御の「中枢」です。

## ファイル別の役割と得意なこと

### `main.py`

- Slack Bot のイベントハンドラ実装
  - `@slack_app.event("app_mention")` でメンションを受け取り、  
    検索 → プロンプト生成 → LLM 呼び出し → Slack 返信までをつなぐ。
  - `CONVERSATIONS` による簡易会話メモリ（ユーザー + チャンネル + スレッド単位）の管理。
- RAG 検索ロジック
  - `_extract_keywords_for_fallback` / `_keyword_search_docs` によるキーワード検索。
  - `search_similar_docs` で「キーワード優先 + ベクトル検索補完」を行う。
  - 英数字の表記揺れ（`deep-wiki` / `DeepWiki` など）を `_normalize_ascii` で吸収。
- LLM 呼び出し
  - `call_llm` で Ollama `/api/generate` をストリーミング呼び出しし、  
    行ごとの JSON チャンクを結合して最終レスポンスを構成。
  - `LLM_NUM_PREDICT` により、環境変数から出力トークン数を調整可能。
- プロンプト設計
  - `build_prompt` で `<chat_history>` + `<context>` + `<question>` を含む指示付きプロンプトを組み立てる。

### `ingest_docs.py`

- `docs/` 配下のファイルをチャンク化し、Ollama embeddings API でベクトル化して `documents` テーブルに保存。
- `split_into_chunks` による文字数ベースのチャンク分割（長文対応）。
- `embed_text` による埋め込み呼び出しとフォールバック
  - 長い入力で 4xx/5xx が返る場合に再帰的分割・平均を行い、「どんな長さの文章でも極力埋め込みを取得する」。
- `upsert_document` で `title / content / embedding / metadata` を1チャンク単位で登録。

## 作業パターン（どんな時にどう動くか）

- 「検索精度を上げたい」
  - まず `search_similar_docs` のキーワード検索部分を確認し、  
    キーワード抽出ルールや `k` の値を調整する。
  - そのうえで、`docs/` 側に FAQ を追加するなど、インデックス元の改善も検討する。
- 「会話のラリーを自然にしたい」
  - `CONVERSATIONS` の保持単位（ユーザー/チャンネル/スレッド）と履歴の長さを調整する。
  - `build_prompt` 内の `<chat_history>` のフォーマットや指示文を変更し、一貫性を高める。
- 「長文をきちんと出力したい」
  - `.env` の `LLM_NUM_PREDICT` を調整し、出力量を増減させる。
  - `_send_long_message` の分割ロジック（`max_len` や改行分割の方針）を必要に応じて変更する。

## 作業上の注意

- DB スキーマ（`sql/init.sql`）と矛盾する変更（embedding 次元数など）は行わないか、行う場合は必ず SQL 側と README を更新する。
- Slack へのエラーメッセージは、日本語でユーザーにわかりやすく返す（スタックトレースは出さない）。
- 例外処理ではログ（`print`）とユーザー向けメッセージを分ける。
- パフォーマンスに効く変更（クエリ回数、チャンク数、会話履歴の長さなど）は、処理フロー全体を確認したうえで行う。
