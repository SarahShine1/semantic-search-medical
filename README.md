# MedSearch AI

A modern, minimalist medical semantic search application powered by advanced AI embeddings and vector databases. Search medical Q&A data using natural language with multiple embedding models and search strategies.

## Overview

MedSearch AI provides a fast, intuitive interface for searching through medical knowledge bases using semantic understanding. It supports both semantic search (understanding meaning) and keyword search, with comparison features to analyze result overlaps between different search methods.

**Key Technologies:**
- **Streamlit** - Modern, responsive web interface
- **PostgreSQL + pgvector** - Vector database for semantic search
- **Sentence Transformers** - State-of-the-art embedding models
- **Docker** - Containerized deployment

---


## Architecture

```
Medical Data (MedQuAD)
        ↓
   [Data Processing]
        ↓
  [Embedding Models]
   ├─ MiniLM (384-dim)
   └─ PubMedBERT (768-dim)
        ↓
PostgreSQL + pgvector
(Vector + Full-text Indexes)
        ↓
    [Streamlit UI]
  ├─ Semantic Search
  ├─ Keyword Search
  └─ Comparison Tools
```

**Database Schema:**
- `medical_documents_minilm` - Documents with MiniLM embeddings (384D)
- `medical_documents_pubmed` - Documents with PubMedBERT embeddings (768D)
- Both tables include: question, answer, category, embedding, full-text indexes

---

## Requirements

- Python 3.9+
- PostgreSQL 12+ with pgvector extension
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for embeddings

---

## Installation

### Option 1: Local Installation

**1. Clone the repository**
```bash
git clone <repository-url>
cd semantic-search-medical
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up PostgreSQL**

Make sure PostgreSQL is running with pgvector extension:

```bash
# Using Docker (recommended)
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=semantic_search_db \
  -p 5433:5432 \
  pgvector/pgvector:pg15

# Then initialize the database
psql -h localhost -p 5433 -U postgres -d semantic_search_db < sql/setup.sql
```

**5. Configure environment variables**

Create a `.env` file:
```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=semantic_search_db
DB_USER=postgres
DB_PASSWORD=postgres
```

**6. Prepare data (if needed)**

```bash
# Download MedQuAD dataset
python download_dataset.py

# Generate embeddings
python src/generate_dual_embeddings.py

# Insert embeddings into database
python src/insert_dual_models.py
```

**7. Run the application**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Option 2: Docker Installation

**1. Using docker-compose (recommended)**
```bash
docker-compose up --build
```

This will:
- Start PostgreSQL with pgvector
- Set up the database
- Run the Streamlit application

The app will be available at `http://localhost:8501`

**2. Manual Docker build**
```bash
docker build -t medsearch-ai .
docker run -p 8501:8501 --env-file .env medsearch-ai
```

---

## Configuration

Edit `config.py` to customize:

```python
class Config:
    # Database settings
    DB_HOST = 'localhost'
    DB_PORT = '5433'
    DB_NAME = 'semantic_search_db'
    
    # Model settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    # Search
    TOP_K_RESULTS = 5  # Number of results to return
    
    # Batch sizes
    DB_INSERT_BATCH_SIZE = 100
    EMBEDDING_BATCH_SIZE = 32
```

---

## Usage

### Running Searches

1. **Enter a medical question** in the search box
2. **Select search method:**
   - "Recherche Rapide" (Fast Search) - Uses MiniLM model
   - "Recherche Médicale" (Medical Search) - Uses PubMedBERT model
   - "Recherche par Mots-Clés" (Keyword Search) - Full-text search
3. **View results** with similarity scores and source information
4. **Compare results** using comparison buttons
5. **Analyze overlap** to see which documents appear in multiple search results

### Interpreting Results

- **Similarity Score** - Ranges from 0-1 (1 = most similar)
- **Chevauchement Badge** - Orange badge indicating result appears in multiple searches
- **Category** - Medical category of the document
- **Source** - Where the data came from (e.g., MedQuAD)

---

## Project Structure

```
semantic-search-medical/
├── app.py                          # Main Streamlit application
├── app_final.py                    # Alternative/backup app version
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Docker composition
├── Dockerfile                      # Docker build file
├── README.md                       # This file
│
├── sql/
│   └── setup.sql                  # Database schema & indexes
│
├── data/
│   ├── raw/
│   │   └── medquad_raw.csv       # Original MedQuAD dataset
│   └── processed/
│       └── medquad_processed.csv # Processed/cleaned data
│
├── embeddings/
│   ├── medquad_embeddings_minilm.npy      # MiniLM embeddings (384D)
│   └── medquad_embeddings_pubmed.npy      # PubMedBERT embeddings (768D)
│
├── src/
│   ├── data_preprocessing.py           # Data cleaning & processing
│   ├── generate_dual_embeddings.py     # Generate embeddings
│   ├── insert_dual_models.py           # Insert into database
│   ├── search_engine.py                # Search logic
│   └── compare_models.py               # Model comparison utilities
│
├── tests/
│   └── [Test files]
│
└── __pycache__/
    └── [Python cache files]
```

---

## Embedding Models

### Model 1: MiniLM (Fast Search)
- **Name:** `all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Inference Speed:** Very fast (< 100ms per query)
- **Use Case:** General medical search, quick results
- **Table:** `medical_documents_minilm`

### Model 2: PubMedBERT (Medical Search)
- **Name:** `pritamdeka/S-PubMedBert-MS-MARCO`
- **Dimensions:** 768
- **Inference Speed:** Fast (< 200ms per query)
- **Use Case:** Specialized medical terminology, better semantic understanding
- **Table:** `medical_documents_pubmed`
- **Trained On:** PubMed medical literature

---


## Troubleshooting

### PostgreSQL Connection Error
```
Error: could not translate host name "localhost" to address
```
**Solution:** Ensure PostgreSQL is running and credentials are correct in `.env`

### No Results Found
- Check database has data (run `python src/insert_dual_models.py`)
- Verify full-text search index exists (`sql/setup.sql` was executed)
- Try different search terms

### Slow Search
- Increase PostgreSQL `work_mem` setting
- Rebuild vector indexes with `CREATE INDEX ... WITH (lists = 200)`
- Use keyword search instead of semantic (usually faster)

### CUDA/GPU Issues
- Models will auto-fallback to CPU
- For GPU acceleration: `pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/`

---

## Performance Benchmarks

On a standard machine (4GB RAM):
- **Semantic Search (Fast):** 80-120ms per query
- **Semantic Search (Medical):** 150-200ms per query
- **Keyword Search:** 30-80ms per query
- **Database Inserts:** ~5,000 docs/second with batch processing

---

## Development

### Adding a New Embedding Model

1. Create new model config in `config.py`:
```python
class Model3Config:
    NAME = 'model-name/from-huggingface'
    DIMENSIONS = 768
    TABLE_NAME = 'medical_documents_model3'
    EMBEDDINGS_FILE = 'embeddings/medquad_embeddings_model3.npy'
```

2. Create table in `sql/setup.sql`:
```sql
CREATE TABLE IF NOT EXISTS medical_documents_model3 (
    ...
    embedding VECTOR(768) NOT NULL,
    ...
);
```

3. Generate embeddings and insert into database

4. Add search option in `app.py` UI

### Running Tests
```bash
pytest tests/
```

---


This project is developed for ESI 2025-2026.

**Author:** BOUCHAMA Sarra

If using publicly, include attribution and follow the terms of dependencies:
- Sentence Transformers: Apache 2.0
- PostgreSQL: PostgreSQL License
- Streamlit: Elastic License

---


## Useful Links

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MedQuAD Dataset](https://github.com/abachaa/MedQuAD)

---


**Last Updated:** February 2026  
**Status:** Production Ready
