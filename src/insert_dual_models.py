"""
Insère les embeddings des 2 modèles dans des tables séparées
"""
import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, Model1Config, Model2Config


def create_table(conn, Config):
    """Crée une table pour un modèle"""
    cursor = conn.cursor()
    
    print(f"\n📊 Création table: {Config.TABLE_NAME}")
    
    # Supprimer si existe
    cursor.execute(f"DROP TABLE IF EXISTS {Config.TABLE_NAME} CASCADE;")
    
    # Créer la table
    cursor.execute(f"""
        CREATE TABLE {Config.TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            combined_text TEXT NOT NULL,
            category VARCHAR(100),
            qtype VARCHAR(50),
            source VARCHAR(200) DEFAULT 'MedQuAD',
            embedding VECTOR({Config.DIMENSIONS}) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Index vectoriel
    print(f"   📌 Création index vectoriel...")
    cursor.execute(f"""
        CREATE INDEX idx_{Config.TABLE_NAME}_embedding 
        ON {Config.TABLE_NAME} 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    
    # Index catégorie
    cursor.execute(f"""
        CREATE INDEX idx_{Config.TABLE_NAME}_category 
        ON {Config.TABLE_NAME}(category);
    """)
    
    conn.commit()
    cursor.close()
    
    print(f"   ✅ Table créée!")


def insert_data(conn, Config, df, embeddings):
    """Insère les données pour un modèle"""
    cursor = conn.cursor()
    
    print(f"\n💾 Insertion dans: {Config.TABLE_NAME}")
    
    # Préparer les données
    data = []
    for idx, row in df.iterrows():
        embedding_list = embeddings[idx].tolist()
        category = row.get('category', 'Unknown')
        
        data.append((
            row['question'],
            row['answer'],
            row['combined_text'],
            category,
            category,  # qtype = category
            'MedQuAD',
            embedding_list
        ))
    
    # Insérer par batch
    insert_query = f"""
        INSERT INTO {Config.TABLE_NAME}
        (question, answer, combined_text, category, qtype, source, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size), desc="Insertion"):
        batch = data[i:i+batch_size]
        execute_batch(cursor, insert_query, batch, page_size=batch_size)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(data)} documents insérés!")


def main():
    print("="*70)
    print("💾 INSERTION DES 2 MODÈLES DANS POSTGRESQL")
    print("="*70)
    
    # Vérifier fichiers
    csv_path = os.path.join(Config.PROCESSED_DATA_DIR, 'medquad_processed.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV non trouvé: {csv_path}")
        return
    
    if not os.path.exists(Model1Config.EMBEDDINGS_FILE):
        print(f"❌ Embeddings modèle 1 non trouvés")
        print(f"💡 Lance: python generate_dual_embeddings.py")
        return
    
    if not os.path.exists(Model2Config.EMBEDDINGS_FILE):
        print(f"❌ Embeddings modèle 2 non trouvés")
        print(f"💡 Lance: python generate_dual_embeddings.py")
        return
    
    # Charger données
    print(f"\n📂 Chargement des données...")
    df = pd.read_csv(csv_path)
    emb1 = np.load(Model1Config.EMBEDDINGS_FILE)
    emb2 = np.load(Model2Config.EMBEDDINGS_FILE)
    
    print(f"   ✅ CSV: {len(df)} documents")
    print(f"   ✅ Embeddings 1: {emb1.shape}")
    print(f"   ✅ Embeddings 2: {emb2.shape}")
    
    # Connexion
    print(f"\n🔌 Connexion PostgreSQL...")
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    # Activer pgvector
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cursor.close()
    
    try:
        # Modèle 1
        print(f"\n{'#'*70}")
        print(f"# MODÈLE 1: {Model1Config.NAME}")
        print(f"{'#'*70}")
        create_table(conn, Model1Config)
        insert_data(conn, Model1Config, df, emb1)
        
        # Modèle 2
        print(f"\n{'#'*70}")
        print(f"# MODÈLE 2: {Model2Config.NAME}")
        print(f"{'#'*70}")
        create_table(conn, Model2Config)
        insert_data(conn, Model2Config, df, emb2)
        
        print(f"\n{'='*70}")
        print("🎉 INSERTION TERMINÉE!")
        print("="*70)
        print(f"\n✅ Les 2 modèles sont prêts pour comparaison!")
        print(f"\n📊 Tables créées:")
        print(f"   - {Model1Config.TABLE_NAME}")
        print(f"   - {Model2Config.TABLE_NAME}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()