-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Tables
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id TEXT NOT NULL,
  document_name TEXT NOT NULL,
  company_name TEXT,
  document_type TEXT,
  document_version TEXT,
  report_year INT,
  report_quarter TEXT,
  file_sha256 TEXT,
  text_sha256 TEXT,
  status TEXT DEFAULT 'INGESTING',
  created_at TIMESTAMPTZ DEFAULT now(),
  CONSTRAINT unique_client_file_sha UNIQUE (client_id, file_sha256),
  CONSTRAINT unique_client_text_sha UNIQUE (client_id, text_sha256)
);

CREATE TABLE IF NOT EXISTS document_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_text TEXT NOT NULL,
  raw_content TEXT,
  page_number INT,
  embedding VECTOR(1536),
  search_tsv tsvector
);

-- Indexes + search vectors
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
  ON document_chunks
  USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS document_chunks_search_tsv_idx
  ON document_chunks 
  USING GIN (search_tsv);

CREATE OR REPLACE FUNCTION document_chunks_generate_tsvector()
RETURNS trigger AS $$
DECLARE
  doc_name TEXT;
  comp_name TEXT;
  doc_ver TEXT;
  rep_year INT;
  rep_quarter TEXT;
BEGIN
  SELECT document_name, company_name, document_version, report_year, report_quarter
  INTO doc_name, comp_name, doc_ver, rep_year, rep_quarter
  FROM documents WHERE id = NEW.document_id;
  
  NEW.search_tsv := 
    setweight(to_tsvector('english', COALESCE(comp_name, '') || ' ' || COALESCE(doc_name, '') || ' ' || COALESCE(doc_ver, '') || ' ' || COALESCE(rep_year::TEXT, '') || ' ' || COALESCE(rep_quarter, '')), 'A') || 
    setweight(to_tsvector('english', COALESCE(NEW.chunk_text, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(NEW.raw_content, '')), 'C');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_document_chunks_tsvector_update
BEFORE INSERT OR UPDATE ON document_chunks
FOR EACH ROW EXECUTE FUNCTION document_chunks_generate_tsvector();

CREATE OR REPLACE FUNCTION documents_propagate_name_change()
RETURNS trigger AS $$
DECLARE
  doc_name TEXT;
  comp_name TEXT;
  doc_ver TEXT;
  rep_year INT;
  rep_quarter TEXT;
BEGIN
  IF OLD.document_name IS DISTINCT FROM NEW.document_name
    OR OLD.company_name IS DISTINCT FROM NEW.company_name
    OR OLD.document_version IS DISTINCT FROM NEW.document_version
    OR OLD.report_year IS DISTINCT FROM NEW.report_year
    OR OLD.report_quarter IS DISTINCT FROM NEW.report_quarter THEN
    SELECT document_name, company_name, document_version, report_year, report_quarter
    INTO doc_name, comp_name, doc_ver, rep_year, rep_quarter
    FROM documents WHERE id = NEW.id;

    UPDATE document_chunks c
    SET search_tsv =
      setweight(to_tsvector('english', COALESCE(comp_name, '') || ' ' || COALESCE(doc_name, '') || ' ' || COALESCE(doc_ver, '') || ' ' || COALESCE(rep_year::TEXT, '') || ' ' || COALESCE(rep_quarter, '')), 'A') ||
      setweight(to_tsvector('english', COALESCE(c.chunk_text, '')), 'B') ||
      setweight(to_tsvector('english', COALESCE(c.raw_content, '')), 'C')
    WHERE c.document_id = NEW.id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_documents_name_change
AFTER UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION documents_propagate_name_change();

-- Hybrid search RPC (vector + keyword + RRF)
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1536),
  user_query text,
  match_threshold float,
  match_count int,
  filter_client_id text,
  filter_years int[] default null,
  filter_quarters text[] default null,
  filter_companies text[] default null,
  filter_document_types text[] default null
)
RETURNS TABLE (
  id UUID,
  document_name TEXT,
  document_version TEXT,
  company_name TEXT,
  document_type TEXT,
  report_year INT,
  report_quarter TEXT,
  created_at TIMESTAMPTZ,
  chunk_text TEXT,
  raw_content TEXT,
  page_number INT,
  similarity FLOAT,
  rrf_score FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_variable
BEGIN
  RETURN QUERY
  WITH vector_matches AS (
    SELECT
      c.id,
      1 - (c.embedding <=> query_embedding) AS similarity,
      ROW_NUMBER() OVER(ORDER BY c.embedding <=> query_embedding) AS rank_vector
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE d.client_id = filter_client_id
      AND (filter_years IS NULL OR d.report_year = ANY(filter_years))
      AND (filter_quarters IS NULL OR d.report_quarter = ANY(filter_quarters))
      AND (filter_companies IS NULL OR d.company_name = ANY(filter_companies))
      AND (filter_document_types IS NULL OR d.document_type = ANY(filter_document_types))
      AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT 100
  ),
  keyword_matches AS (
    SELECT
      c.id,
      ts_rank_cd(c.search_tsv, plainto_tsquery('english', user_query)) AS rank_score,
      ROW_NUMBER() OVER(ORDER BY ts_rank_cd(c.search_tsv, plainto_tsquery('english', user_query)) DESC) AS rank_keyword
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE d.client_id = filter_client_id
      AND (filter_years IS NULL OR d.report_year = ANY(filter_years))
      AND (filter_quarters IS NULL OR d.report_quarter = ANY(filter_quarters))
      AND (filter_companies IS NULL OR d.company_name = ANY(filter_companies))
      AND (filter_document_types IS NULL OR d.document_type = ANY(filter_document_types))
      AND (user_query IS NULL OR c.search_tsv @@ plainto_tsquery('english', user_query))
    ORDER BY rank_score DESC
    LIMIT 100
  )
  SELECT
    COALESCE(v.id, k.id) AS id,
    d.document_name,
    d.document_version,
    d.company_name,
    d.document_type,
    d.report_year, 
    d.report_quarter,
    d.created_at,
    c.chunk_text,
    c.raw_content,
    c.page_number,
    COALESCE(v.similarity, 0.0::FLOAT) AS similarity,
    ((CASE WHEN v.rank_vector IS NOT NULL THEN 1.0 / (60 + v.rank_vector) ELSE 0 END) +
    (CASE WHEN k.rank_keyword IS NOT NULL THEN 1.0 / (60 + k.rank_keyword) ELSE 0 END))::FLOAT AS rrf_score
  FROM vector_matches v
  FULL OUTER JOIN keyword_matches k ON v.id = k.id
  JOIN document_chunks c ON c.id = COALESCE(v.id, k.id)
  JOIN documents d ON c.document_id = d.id
  ORDER BY rrf_score DESC
  LIMIT match_count;
END;
$$;

-- Permissions
GRANT ALL ON public.documents TO service_role;
GRANT ALL ON public.document_chunks TO service_role;
