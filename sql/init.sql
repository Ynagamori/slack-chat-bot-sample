CREATE EXTENSION IF NOT EXISTS vector;

-- nomic-embed-text は 768 次元なので 768 を指定
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  title TEXT,
  content TEXT,
  embedding VECTOR(768),
  metadata JSONB
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
