"""
Compare les 2 modèles côte à côte
"""
import time
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, Model1Config, Model2Config


class DualModelComparer:
    def __init__(self):
        print("🔧 Initialisation...")
        
        # Charger les 2 modèles
        print(f"   📦 Modèle 1: {Model1Config.NAME}")
        self.model1 = SentenceTransformer(Model1Config.NAME)
        
        print(f"   📦 Modèle 2: {Model2Config.NAME}")
        self.model2 = SentenceTransformer(Model2Config.NAME)
        
        # Connexion DB
        self.conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        
        print("   ✅ Prêt!\n")
    
    def search_model1(self, query, top_k=5):
        """Recherche avec modèle 1"""
        # Encoder
        emb = self.model1.encode(query, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)
        
        # Rechercher
        cursor = self.conn.cursor()
        start = time.time()
        
        cursor.execute(f"""
            SELECT id, question, category, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM {Model1Config.TABLE_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb.tolist(), emb.tolist(), top_k))
        
        results = cursor.fetchall()
        search_time = time.time() - start
        cursor.close()
        
        return results, search_time
    
    def search_model2(self, query, top_k=5):
        """Recherche avec modèle 2"""
        # Encoder
        emb = self.model2.encode(query, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)
        
        # Rechercher
        cursor = self.conn.cursor()
        start = time.time()
        
        cursor.execute(f"""
            SELECT id, question, category, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM {Model2Config.TABLE_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb.tolist(), emb.tolist(), top_k))
        
        results = cursor.fetchall()
        search_time = time.time() - start
        cursor.close()
        
        return results, search_time
    
    def compare(self, query, top_k=5):
        """Compare les 2 modèles"""
        print("="*80)
        print(f"🔍 COMPARAISON: '{query}'")
        print("="*80)
        
        # Modèle 1
        print(f"\n🔷 MODÈLE 1: {Model1Config.NAME}")
        print(f"   {Model1Config.DESCRIPTION}")
        print("-"*80)
        
        r1, t1 = self.search_model1(query, top_k)
        print(f"⏱️  Temps: {t1*1000:.2f}ms\n")
        
        for i, (doc_id, question, category, sim) in enumerate(r1, 1):
            print(f"  {i}. [{category}] {question[:70]}...")
            print(f"     Similarité: {sim:.4f}\n")
        
        # Modèle 2
        print(f"\n🔶 MODÈLE 2: {Model2Config.NAME}")
        print(f"   {Model2Config.DESCRIPTION}")
        print("-"*80)
        
        r2, t2 = self.search_model2(query, top_k)
        print(f"⏱️  Temps: {t2*1000:.2f}ms\n")
        
        for i, (doc_id, question, category, sim) in enumerate(r2, 1):
            print(f"  {i}. [{category}] {question[:70]}...")
            print(f"     Similarité: {sim:.4f}\n")
        
        # Analyse
        print("\n📊 ANALYSE")
        print("-"*80)
        print(f"⏱️  Performance:")
        print(f"   Modèle 1: {t1*1000:.2f}ms")
        print(f"   Modèle 2: {t2*1000:.2f}ms")
        
        ids1 = {r[0] for r in r1}
        ids2 = {r[0] for r in r2}
        overlap = len(ids1 & ids2)
        
        print(f"\n🔄 Chevauchement: {overlap}/{top_k} documents communs")
        
        avg_sim1 = np.mean([r[3] for r in r1])
        avg_sim2 = np.mean([r[3] for r in r2])
        
        print(f"\n⭐ Similarité moyenne:")
        print(f"   Modèle 1: {avg_sim1:.4f}")
        print(f"   Modèle 2: {avg_sim2:.4f}")
        
        print("\n" + "="*80 + "\n")
    
    def close(self):
        self.conn.close()


def main():
    comparer = DualModelComparer()
    
    test_queries = [
        "How to treat diabetes?",
        "What are the symptoms of heart disease?",
        "Is cancer hereditary?",
        "What causes high blood pressure?",
        "How is Alzheimer's diagnosed?"
    ]
    
    print(f"🧪 SUITE DE TESTS ({len(test_queries)} requêtes)\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*80}")
        print(f"# TEST {i}/{len(test_queries)}")
        print(f"{'#'*80}\n")
        
        comparer.compare(query, top_k=3)
        
        if i < len(test_queries):
            input("⏸️  Enter pour continuer...")
    
    comparer.close()
    print("✅ Terminé!")


if __name__ == "__main__":
    main()