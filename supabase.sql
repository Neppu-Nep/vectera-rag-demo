-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the table to store document-level metadata
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id TEXT NOT NULL,          -- Client multi-tenant segregation
  document_name TEXT NOT NULL,      -- e.g., "Apple_Q3_2023.pdf"
  company_name TEXT,                -- The extracted company name
  document_version TEXT,            -- To handle the versioning/conflict trap
  file_sha256 TEXT,                 -- For deduplication
  text_sha256 TEXT                  -- For deduplication
);

-- 3. Create the table to store chunks and embeddings
CREATE TABLE document_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_text TEXT NOT NULL,         -- The actual chunk of text
  embedding VECTOR(3072)            -- OpenAI text-embedding-3-large uses 3072 dimensions
);

-- 4. Create the Semantic Search function (Cosine Similarity)
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(3072),
  match_threshold FLOAT,
  match_count INT,
  filter_client_id TEXT
)
RETURNS TABLE (
  id UUID,
  document_name TEXT,
  document_version TEXT,
  company_name TEXT,
  chunk_text TEXT,
  similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    c.id,
    d.document_name,
    d.document_version,
    d.company_name,
    c.chunk_text,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM document_chunks c
  JOIN documents d ON c.document_id = d.id
  WHERE d.client_id = filter_client_id
  AND 1 - (c.embedding <=> query_embedding) > match_threshold
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- 5. Give permissions to the service role
GRANT ALL ON public.documents TO service_role;
GRANT ALL ON public.document_chunks TO service_role;