import os
import json
import itertools
import time
import re
import requests
import psycopg2
from slack_bolt import App as SlackApp
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ====== 環境変数 ======
DB_HOST = os.getenv("DB_HOST") or "pgvector-db"
DB_USER = os.getenv("DB_USER") or "llm"
DB_PASSWORD = os.getenv("DB_PASSWORD") or "llm_pass"
DB_NAME = os.getenv("DB_NAME") or "llmdb"

OLLAMA_URL = "http://ollama:11434"
LLM_MODEL = os.getenv("LLM_MODEL") or "llama3.2:3b-instruct-q4_0"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "nomic-embed-text"
try:
    LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT") or "1024")
except ValueError:
    LLM_NUM_PREDICT = 512

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

# ====== 簡易な会話メモリ（プロセス内） ======
# キー: "<user>:<channel>:<thread_ts|root>"
# 値: [{"role": "user"|"assistant", "content": str}, ...]
CONVERSATIONS: dict[str, list[dict[str, str]]] = {}

# ====== Ollama ヘルスチェック ======
def wait_for_ollama(max_attempts: int = 20, base_wait: float = 1.5):
    """Ollama APIが応答するまでポーリングする。"""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                return
            raise RuntimeError(f"Ollama not ready: status {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = base_wait * attempt
            print(f"Ollama not ready (attempt {attempt}/{max_attempts}): {exc}")
            if attempt == max_attempts:
                break
            time.sleep(wait)
    raise last_exc

# 起動時にOllamaが立ち上がるまで待つ
wait_for_ollama()

# ====== DB 接続 ======
def connect_with_retry(max_attempts: int = 10, base_wait: float = 2.0):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return psycopg2.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                dbname=DB_NAME,
            )
        except psycopg2.OperationalError as exc:
            last_exc = exc
            wait = base_wait * attempt
            print(f"DB connection failed (attempt {attempt}/{max_attempts}): {exc}")
            if attempt == max_attempts:
                break
            time.sleep(wait)
    raise last_exc


conn = connect_with_retry()

def get_conn():
    """再利用中のコネクションが落ちていたら張り直す。"""
    global conn
    if conn.closed:
        conn = connect_with_retry()
    return conn

def _call_embeddings_api(text: str):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # {"embedding": [...]} を想定
    return data["embedding"]


def embed_text(text: str, depth: int = 0, max_depth: int = 4):
    """Ollama embeddings API への安全なラッパー。

    長い入力で 4xx/5xx が返る場合、テキストを分割してサブチャンクの
    ベクトルを平均し、1つのベクトルとして返す。
    """
    try:
        return _call_embeddings_api(text)
    except requests.HTTPError as exc:
        resp = exc.response
        status = resp.status_code if resp is not None else None

        if status and 400 <= status < 600 and depth < max_depth and len(text) > 50:
            mid = len(text) // 2
            left = text[:mid].strip()
            right = text[mid:].strip()
            parts = [p for p in (left, right) if p]
            vecs = []
            for part in parts:
                vecs.append(embed_text(part, depth=depth + 1, max_depth=max_depth))

            if not vecs:
                body = resp.text if resp is not None else ""
                raise RuntimeError(f"Embeddings API error ({exc}): {body}") from exc

            dim = len(vecs[0])
            for v in vecs[1:]:
                if len(v) != dim:
                    dim = min(dim, len(v))
            if dim == 0:
                body = resp.text if resp is not None else ""
                raise RuntimeError(f"Embeddings API error ({exc}): {body}") from exc

            agg = [0.0] * dim
            for v in vecs:
                for i in range(dim):
                    agg[i] += float(v[i])
    return [x / len(vecs) for x in agg]

# ====== ベクトル検索 ======
def _extract_keywords_for_fallback(text: str):
    """ベクトル検索がうまくいかなかった場合のための簡易キーワード抽出。"""
    # よく出る記号や助詞をスペースに置き換えてラフに分割
    separators = [
        "　",
        " ",
        "\n",
        "\t",
        "、",
        "。",
        "？",
        "?",
        "！",
        "!",
        "「",
        "」",
        "『",
        "』",
        "（",
        "）",
        "(",
        ")",
        "・",
        "：",
        ":",
        "〜",
        "～",
        "　",
    ]
    particles = ["の", "を", "が", "は", "に", "で", "と", "も", "へ", "から", "まで", "より"]

    tmp = text
    for ch in separators + particles:
        tmp = tmp.replace(ch, " ")

    rough_tokens = [t.strip() for t in tmp.split(" ") if t.strip()]

    keywords = set()
    for token in rough_tokens:
        if len(token) < 2:
            continue

        # 英数字だけのトークン（Wi-Fiなど）はそのまま
        if all(ord(c) < 128 for c in token):
            keywords.add(token.lower())
            continue

        # 日本語などのトークンは2〜6文字程度の部分文字列をキーワードとして使う
        max_len = min(6, len(token))
        for size in range(2, max_len + 1):
            for i in range(0, len(token) - size + 1):
                keywords.add(token[i : i + size])

    return keywords


def _normalize_ascii(text: str) -> str:
    """英数字トークンを類似ワードとして扱うための正規化。

    - 小文字化
    - 英数字以外（ハイフン、アンダースコア、ピリオドなど）を除去
    例: \"DeepWiki\" / \"deep-wiki\" / \"deep wiki\" -> \"deepwiki\"
    """
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _keyword_search_docs(query: str, k: int = 3):
    """ベクトル検索でヒットしなかった場合のフォールバック用: 全文を読み込んで素朴なキーワードマッチでスコア付け。"""
    keywords = _extract_keywords_for_fallback(query)
    if not keywords:
        return []

    with get_conn().cursor() as cur:
        cur.execute("SELECT title, content FROM documents")
        rows = cur.fetchall()

    scored = []
    for title, content in rows:
        text = f"{title or ''}\n{content or ''}"
        text_lower = text.lower()
        text_ascii = _normalize_ascii(text_lower)
        score = 0
        for kw in keywords:
            if all(ord(c) < 128 for c in kw):
                # deep-wiki / DeepWiki / deep wiki などの揺れを吸収
                if kw in text_ascii:
                    score += len(kw)
            else:
                if kw in text:
                    score += len(kw)

        if score > 0:
            scored.append((score, {"title": title, "content": content}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


def search_similar_docs(query: str, k: int = 3):
    """キーワードベース検索を優先し、必要に応じてベクトル検索で補完する。"""
    # 1. まず素朴なキーワード検索で「それらしい」ドキュメントを拾う
    docs = _keyword_search_docs(query, k=k)
    existing_titles = {d["title"] for d in docs}

    # 2. まだ枠に余裕があれば、ベクトル検索で補完（同じタイトルは除外）
    remaining = k - len(docs)
    if remaining > 0:
        try:
            vec = embed_text(query)
            vec_literal = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

            with get_conn().cursor() as cur:
                cur.execute(
                    """
                    SELECT title, content
                    FROM documents
                    ORDER BY embedding <-> %s
                    LIMIT %s
                    """,
                    (vec_literal, k),
                )
                rows = cur.fetchall()

            for title, content in rows:
                if title in existing_titles:
                    continue
                docs.append({"title": title, "content": content})
                existing_titles.add(title)
                if len(docs) >= k:
                    break
        except Exception as exc:  # noqa: BLE001
            try:
                print(f"[RAG] vector search error: {exc}")
            except Exception:
                pass

    # 簡易デバッグログ（検索クエリとヒットしたドキュメント）
    try:
        print(f"[RAG] search_query={query!r}")
        print(f"[RAG] hits={[d['title'] for d in docs]}")
    except Exception:
        # ログで失敗しても検索自体は続行する
        pass

    return docs

def call_llm(prompt: str) -> str:
    """Ollama にストリーミングで問い合わせ、チャンクを結合して返す。"""
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            # 事実ベースでブレない回答を出す
            "temperature": 0.1,
            # 環境変数で調整可能な出力トークン上限
            "num_predict": LLM_NUM_PREDICT,
        },
    }
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()

            chunks: list[str] = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                part = data.get("response")
                if part:
                    chunks.append(part)
                if data.get("done"):
                    break

            text = "".join(chunks).strip()
            if not text:
                # 念のため、単発JSONレスポンス形式もフォロー
                try:
                    fallback = r.json()
                    return (fallback.get("response") or "").strip()
                except Exception:
                    return "社内LLMから有効な応答を取得できませんでした。少し時間をおいて、もう一度お試しください。"
            return text
    except requests.exceptions.ReadTimeout:
        # モデル応答が遅すぎる場合は Slack にスタックトレースを出さずにメッセージだけ返す
        return "社内LLMからの応答に時間がかかりすぎたためタイムアウトしました。少し時間をおいて、もう一度お試しください。"
    except requests.RequestException as exc:
        return f"社内LLM呼び出し中にエラーが発生しました: {exc}"


def build_search_query(user_question: str) -> str:
    """クエリリライトを行わず、そのまま検索に使う。

    ※将来的に会話履歴を使ったリライトを行う場合は、この関数を拡張する。
    """
    return user_question

def build_prompt(user_question: str, docs, chat_history=None):
    """RAGコンテキストと会話履歴を含んだプロンプトを組み立てる。"""

    # 会話履歴（直近のやり取りのみ使用）
    history_items = chat_history or []
    recent_history = history_items[-10:]  # 最大 10 メッセージ分
    history_parts: list[str] = []
    for i, turn in enumerate(recent_history, 1):
        role = "ユーザー" if turn.get("role") == "user" else "アシスタント"
        content = turn.get("content", "")
        history_parts.append(f"[TURN{i}] {role}: {content}")
    history_text = "\n".join(history_parts) if history_parts else "（直近の会話履歴はありません）"

    # RAG コンテキスト
    context_parts = []
    for i, d in enumerate(docs, 1):
        context_parts.append(f"[DOC{i}] {d['title']}\n{d['content']}\n")
    context = "\n\n".join(context_parts) if context_parts else "（関連ドキュメントは見つかりませんでした）"

    prompt = f"""
あなたは、提供されたコンテキスト（文脈情報）および会話履歴に基づいて回答する、誠実で厳格なAIアシスタントです。
以下の `<chat_history>` と `<context>` 内の情報を唯一の真実として扱い、`<question>` タグ内の質問に答えてください。

# 指示事項
1. **情報源の限定**: 自身の一般的な知識は極力使用せず、`<chat_history>` と `<context>` に書かれている内容を優先してください。そこにない事柄は「文脈情報に含まれていないため、わかりません」と回答してください。
2. **忠実な引用**: 数値、パスワード、パス、URL、コマンド等の具体的な値は、変更・省略・要約せず、記載通りに正確に出力してください。
3. **形式の遵守**: ユーザーが回答形式（箇条書き、行数など）を指定した場合は、その形式を厳守してください。
4. **会話の一貫性**: フォローアップの質問の場合は、会話履歴を踏まえて、前回の回答との一貫性を保ってください。
5. **推測の禁止**: 文脈から論理的に導き出せない場合、無理に回答を捏造しないでください。

# 会話履歴 (Chat History)
<chat_history>
{history_text}
</chat_history>

# 関連ドキュメント (Context)
<context>
{context}
</context>

# ユーザーの質問 (Question)
<question>
{user_question}
</question>

# 回答
上記に基づき、最終的な回答のみを出力してください：
"""
    return prompt.strip()

# ====== Slack Bot ======
slack_app = SlackApp(token=SLACK_BOT_TOKEN)

@slack_app.event("app_mention")
def handle_app_mention(body, say):
    event = body.get("event", {}) or {}
    text = event.get("text", "") or ""
    channel = event.get("channel")
    user_id = event.get("user")
    thread_ts = event.get("thread_ts") or event.get("ts")

    print("--debug--")
    print(body)

    # メンション部分をざっくり削る（先頭トークンを捨てる）
    user_question = " ".join(text.split()[1:]) if " " in text else text

    # 会話キーを決定（ユーザー + チャンネル + スレッド）
    if user_id and channel:
        conv_key = f"{user_id}:{channel}:{thread_ts or 'root'}"
    else:
        conv_key = None
    history = CONVERSATIONS.get(conv_key, []) if conv_key else []

    # 1. 検索用クエリ生成（クエリリライト）
    search_query = build_search_query(user_question)

    # 2. ドキュメント検索（キーワード + ベクトル）
    #    コンテキストを絞って応答を短く・高速にするため k は 3 に抑える
    docs = search_similar_docs(search_query, k=3)

    # 3. プロンプト組み立て（会話履歴込み）
    prompt = build_prompt(user_question, docs, chat_history=history)

    # 4. LLM 呼び出し
    answer = call_llm(prompt)

    # 5. Slack へ返信（スレッド内でラリーできるように thread_ts を指定）
    _send_long_message(say, answer, thread_ts=thread_ts)

    # 6. 会話履歴の更新（直近のみ保持）
    if conv_key:
        new_history = history + [
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": answer},
        ]
        # 最大 10 メッセージ分に制限してメモリを抑える
        if len(new_history) > 10:
            new_history = new_history[-10:]
        CONVERSATIONS[conv_key] = new_history


def _send_long_message(say, text: str, thread_ts: str | None = None, max_len: int = 3500):
    """Slack の文字数制限を考慮しつつ長文を分割送信する。"""
    if not text:
        return

    remaining = text
    first = True
    while remaining:
        if len(remaining) <= max_len:
            chunk = remaining
            remaining = ""
        else:
            # できるだけ改行で分割する
            split_pos = remaining.rfind("\n", 0, max_len)
            if split_pos == -1:
                split_pos = max_len
            chunk = remaining[:split_pos]
            remaining = remaining[split_pos:].lstrip("\n")

        if thread_ts:
            say(chunk, thread_ts=thread_ts)
        else:
            say(chunk)

        first = False

if __name__ == "__main__":
    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()
