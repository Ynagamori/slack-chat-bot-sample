import json
import os
import time
import hashlib
from pathlib import Path

import psycopg2
import requests

# ====== 環境変数 ======
DB_HOST = os.getenv("DB_HOST") or "pgvector-db"
DB_USER = os.getenv("DB_USER") or "llm"
DB_PASSWORD = os.getenv("DB_PASSWORD") or "llm_pass"
DB_NAME = os.getenv("DB_NAME") or "llmdb"

OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "nomic-embed-text"

# 取り込み対象ディレクトリ
DOCS_DIR = Path(os.getenv("DOCS_DIR") or "/app/docs")


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


def split_into_chunks(text: str, max_chars: int = 400):
    """長文テキストをチャンクに分割する。行単位でできるだけ自然に切る。

    NOTE:
    nomic-embed-text の embeddings API は一度に処理できるトークン数の制約があり、
    文字数ベースで大きすぎるチャンクを投げると
    「caching disabled but unable to fit entire input in a batch」
    という panic を起こすことがある。
    日本語を多く含むテキストでは「文字数 ≒ トークン数」とみなして安全側に倒し、
    デフォルトのチャンクサイズを 400 文字に抑えている。
    """
    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        # 1行が極端に長い場合は、そのままでも分割する
        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            # 超過行をそのままチャンクとして追加
            for i in range(0, len(line), max_chars):
                chunks.append(line[i : i + max_chars])
            continue

        if len(current) + len(line) > max_chars:
            chunks.append(current)
            current = ""
        current += line
    if current:
        chunks.append(current)
    return chunks


def _call_embeddings_api(text: str):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["embedding"]


def embed_text(text: str, depth: int = 0, max_depth: int = 4):
    """Ollama embeddings API への安全なラッパー。

    入力が長すぎて 500 エラーになる場合、テキストを再帰的に分割して
    サブチャンクごとのベクトルを平均し、1つのベクトルに統合する。
    これにより、どんな長さの文章でも（モデルの制約内で）できるだけ埋め込みを取得する。
    """
    try:
        return _call_embeddings_api(text)
    except requests.HTTPError as exc:
        resp = exc.response
        status = resp.status_code if resp is not None else None

        # HTTPの 4xx/5xx は、長さ起因のエラーの可能性があるので分割して再試行する
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

            # サブチャンクのベクトルを単純平均で統合
            agg = [0.0] * dim
            for v in vecs:
                for i in range(dim):
                    agg[i] += float(v[i])
            return [x / len(vecs) for x in agg]

        body = resp.text if resp is not None else ""
        raise RuntimeError(f"Embeddings API error ({exc}): {body}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Embeddings API request failed: {exc}") from exc


def iter_doc_files(base_dir: Path):
    exts = {".txt", ".md", ".mdx"}
    for path in base_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def upsert_document(cur, title: str, content: str, embedding, metadata: dict):
    """documents テーブルに 1 レコード挿入する（トランザクションは呼び出し側で管理）。"""
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
    cur.execute(
        """
        INSERT INTO documents (title, content, embedding, metadata)
        VALUES (%s, %s, %s, %s)
        """,
        (title, content, vec_literal, json.dumps(metadata)),
    )


def get_existing_index_map() -> dict[str, str | None]:
    """既存インデックスの path -> content_hash マップを作成する。

    旧バージョンで取り込み済みのレコードには content_hash がないため、
    その場合は値が None になる（要再インデックス扱い）。
    """
    with get_conn().cursor() as cur:
        cur.execute(
            """
            SELECT metadata->>'path', metadata->>'content_hash'
            FROM documents
            WHERE metadata->>'path' IS NOT NULL
            """,
        )
        rows = cur.fetchall()
    return {path: content_hash or None for path, content_hash in rows}


def main():
    if not DOCS_DIR.exists():
        print(f"docsディレクトリが見つかりません: {DOCS_DIR}")
        return

    files = list(iter_doc_files(DOCS_DIR))
    if not files:
        print(f"取り込み対象ファイルがありません: {DOCS_DIR}")
        return

    print(f"{len(files)} 件のドキュメントを取り込みます")

    # 既存インデックス情報を一括取得（差分検知 + 削除検知用）
    existing_index = get_existing_index_map()
    indexed_paths = set(existing_index.keys())

    current_paths: set[str] = set()
    conn = get_conn()
    with conn.cursor() as cur:
        for path in files:
            rel_path = path.relative_to(DOCS_DIR)
            path_str = str(rel_path)
            current_paths.add(path_str)
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                print(f"[SKIP] エンコードできないためスキップ: {rel_path}")
                continue

            # ファイル内容のハッシュを用いて、変更の有無を判定
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            existing_hash = existing_index.get(path_str)
            if existing_hash == content_hash:
                print(f"[SKIP] 変更なし: {rel_path}")
                continue

            # 既存の同一ファイル由来レコードを削除
            cur.execute(
                "DELETE FROM documents WHERE metadata->>'path' = %s",
                (path_str,),
            )

            chunks = split_into_chunks(content)
            print(f"[EMBED] {rel_path} ({len(chunks)} chunks)")

            for idx, chunk in enumerate(chunks, start=1):
                try:
                    vec = embed_text(chunk)
                except Exception as exc:  # noqa: BLE001
                    print(f"[ERROR] embeddings failed: {rel_path} (chunk {idx}): {exc}")
                    print("[HINT] Ollama のログ(ollamaコンテナ)を確認してください。")
                    # このファイルの残りチャンクはスキップし、次のファイルに進む
                    break
                meta = {
                    "path": path_str,
                    "chunk": idx,
                    "content_hash": content_hash,
                }
                title = f"{rel_path}#chunk-{idx}"
                upsert_document(cur, title=title, content=chunk, embedding=vec, metadata=meta)

        # ファイルシステム上から削除されたドキュメントを検知し、DB からも削除
        removed_paths = sorted(indexed_paths - current_paths)
        if removed_paths:
            print(f"[CLEANUP] ファイルが削除されたため、インデックスからも削除します ({len(removed_paths)} paths)")
            for p in removed_paths:
                print(f"[CLEANUP] DELETE path={p}")
                cur.execute(
                    "DELETE FROM documents WHERE metadata->>'path' = %s",
                    (p,),
                )

        conn.commit()

    print("取り込み完了")


if __name__ == "__main__":
    wait_for_ollama()
    main()
