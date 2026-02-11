-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create main table for medical documents
CREATE TABLE medical_documents (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding VECTOR(384),
    category TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector index for similarity search
CREATE INDEX ON medical_documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create text search indexes for comparison
CREATE INDEX idx_question_text ON medical_documents 
USING GIN(to_tsvector('english', question));

CREATE INDEX idx_answer_text ON medical_documents 
USING GIN(to_tsvector('english', answer));

-- Create category index
CREATE INDEX idx_category ON medical_documents(category);