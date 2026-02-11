import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5433')
    DB_NAME = os.getenv('DB_NAME', 'semantic_search_db')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
    
    # Embedding Model
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    # Search
    TOP_K_RESULTS = 5
    
    # Paths
    DATA_DIR = 'data'
    RAW_DATA_DIR = 'data/raw'
    PROCESSED_DATA_DIR = 'data/processed'
    EMBEDDINGS_DIR = 'embeddings'

     # Batch sizes (ADD THESE LINES)
    DB_INSERT_BATCH_SIZE = 100  # For database insertion
    EMBEDDING_BATCH_SIZE = 32   # For embedding generation


class Model1Config:
    """Modèle 1: MiniLM (général, rapide)"""
    NAME = 'all-MiniLM-L6-v2'
    DIMENSIONS = 384
    TABLE_NAME = 'medical_documents_minilm'
    EMBEDDINGS_FILE = 'embeddings/medquad_embeddings_minilm.npy'
    DESCRIPTION = 'Modèle général rapide'


class Model2Config:
    """Modèle 2: PubMedBert (spécialisé médical)"""
    NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'
    DIMENSIONS = 768  # Plus de dimensions!
    TABLE_NAME = 'medical_documents_pubmed'
    EMBEDDINGS_FILE = 'embeddings/medquad_embeddings_pubmed.npy'
    DESCRIPTION = 'Modèle spécialisé médical'