"""
Comparaison de 3 méthodes de recherche:
1. Recherche classique (mots-clés)
2. Recherche sémantique MiniLM (général)
3. Recherche sémantique PubMedBert (médical)
"""
import time
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, Model1Config, Model2Config


class TripleSearchComparer:
    """Compare les 3 méthodes de recherche"""
    
    def __init__(self):
        print("🔧 Initialisation des 3 moteurs de recherche...")
        
        # Charger les 2 modèles sémantiques
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
        
        print("   ✅ Les 3 moteurs sont prêts!\n")
    
    def keyword_search(self, query, top_k=5):
        """
        Méthode 1: Recherche classique par mots-clés
        Utilise PostgreSQL Full-Text Search
        """
        cursor = self.conn.cursor()
        start = time.time()
        
        # Full-text search PostgreSQL
        sql = """
            SELECT 
                id, 
                question, 
                answer,
                category,
                qtype,
                ts_rank(
                    to_tsvector('english', combined_text), 
                    plainto_tsquery('english', %s)
                ) as rank
            FROM medical_documents_minilm
            WHERE to_tsvector('english', combined_text) @@ 
                  plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        
        cursor.execute(sql, (query, query, top_k))
        results = cursor.fetchall()
        search_time = time.time() - start
        cursor.close()
        
        return results, search_time
    
    def semantic_search_model1(self, query, top_k=5):
        """
        Méthode 2: Recherche sémantique MiniLM (général)
        """
        # Encoder avec modèle 1
        emb = self.model1.encode(query, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)
        
        cursor = self.conn.cursor()
        start = time.time()
        
        cursor.execute(f"""
            SELECT 
                id, 
                question, 
                answer,
                category,
                qtype,
                1 - (embedding <=> %s::vector) as similarity
            FROM {Model1Config.TABLE_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb.tolist(), emb.tolist(), top_k))
        
        results = cursor.fetchall()
        search_time = time.time() - start
        cursor.close()
        
        return results, search_time
    
    def semantic_search_model2(self, query, top_k=5):
        """
        Méthode 3: Recherche sémantique PubMedBert (médical)
        """
        # Encoder avec modèle 2
        emb = self.model2.encode(query, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)
        
        cursor = self.conn.cursor()
        start = time.time()
        
        cursor.execute(f"""
            SELECT 
                id, 
                question, 
                answer,
                category,
                qtype,
                1 - (embedding <=> %s::vector) as similarity
            FROM {Model2Config.TABLE_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb.tolist(), emb.tolist(), top_k))
        
        results = cursor.fetchall()
        search_time = time.time() - start
        cursor.close()
        
        return results, search_time
    
    def compare_all(self, query, top_k=5):
        """Compare les 3 méthodes côte à côte"""
        
        print("="*90)
        print(f"🔍 COMPARAISON DES 3 MÉTHODES")
        print("="*90)
        print(f"📝 Requête: '{query}'")
        print("="*90 + "\n")
        
        # 1. Recherche classique
        print("🔷 MÉTHODE 1: RECHERCHE CLASSIQUE (Mots-clés)")
        print("   📖 PostgreSQL Full-Text Search")
        print("-"*90)
        
        r1, t1 = self.keyword_search(query, top_k)
        print(f"⏱️  Temps: {t1*1000:.2f}ms")
        print(f"📊 Résultats: {len(r1)}\n")
        
        if len(r1) == 0:
            print("   ⚠️  Aucun résultat trouvé (aucun mot-clé ne correspond)\n")
        else:
            for i, (doc_id, question, answer, category, qtype, rank) in enumerate(r1, 1):
                print(f"  {i}. [{category}] {question[:70]}...")
                print(f"     Rank: {rank:.4f}\n")
        
        # 2. Sémantique Modèle 1
        print("\n🔶 MÉTHODE 2: RECHERCHE SÉMANTIQUE (Modèle Général)")
        print(f"   📦 {Model1Config.NAME}")
        print("-"*90)
        
        r2, t2 = self.semantic_search_model1(query, top_k)
        print(f"⏱️  Temps: {t2*1000:.2f}ms")
        print(f"📊 Résultats: {len(r2)}\n")
        
        for i, (doc_id, question, answer, category, qtype, sim) in enumerate(r2, 1):
            print(f"  {i}. [{category}] {question[:70]}...")
            print(f"     Similarité: {sim:.4f}\n")
        
        # 3. Sémantique Modèle 2
        print("\n🔷 MÉTHODE 3: RECHERCHE SÉMANTIQUE (Modèle Médical)")
        print(f"   📦 {Model2Config.NAME}")
        print("-"*90)
        
        r3, t3 = self.semantic_search_model2(query, top_k)
        print(f"⏱️  Temps: {t3*1000:.2f}ms")
        print(f"📊 Résultats: {len(r3)}\n")
        
        for i, (doc_id, question, answer, category, qtype, sim) in enumerate(r3, 1):
            print(f"  {i}. [{category}] {question[:70]}...")
            print(f"     Similarité: {sim:.4f}\n")
        
        # 4. Analyse comparative
        print("\n📊 ANALYSE COMPARATIVE")
        print("-"*90)
        
        print(f"\n⏱️  Performance:")
        print(f"   Classique:  {t1*1000:6.2f}ms")
        print(f"   Sémantique 1: {t2*1000:6.2f}ms")
        print(f"   Sémantique 2: {t3*1000:6.2f}ms")
        
        if len(r1) > 0:
            ids1 = {r[0] for r in r1}
            ids2 = {r[0] for r in r2}
            ids3 = {r[0] for r in r3}
            
            overlap_12 = len(ids1 & ids2)
            overlap_13 = len(ids1 & ids3)
            overlap_23 = len(ids2 & ids3)
            overlap_all = len(ids1 & ids2 & ids3)
            
            print(f"\n🔄 Chevauchement des résultats:")
            print(f"   Classique ∩ Sémantique 1: {overlap_12}/{top_k}")
            print(f"   Classique ∩ Sémantique 2: {overlap_13}/{top_k}")
            print(f"   Sémantique 1 ∩ Sémantique 2: {overlap_23}/{top_k}")
            print(f"   Communs aux 3: {overlap_all}/{top_k}")
        
        # Scores moyens
        if len(r2) > 0:
            avg_sim2 = np.mean([r[5] for r in r2])
            avg_sim3 = np.mean([r[5] for r in r3])
            
            print(f"\n⭐ Scores moyens (similarité):")
            print(f"   Sémantique 1 (général): {avg_sim2:.4f}")
            print(f"   Sémantique 2 (médical): {avg_sim3:.4f}")
        
        print("\n" + "="*90 + "\n")
        
        return {
            'keyword': (r1, t1),
            'semantic1': (r2, t2),
            'semantic2': (r3, t3)
        }
    
    def close(self):
        self.conn.close()


def main():
    """Test avec plusieurs requêtes"""
    
    print("🧪 SUITE DE TESTS - COMPARAISON 3 MÉTHODES\n")
    
    comparer = TripleSearchComparer()
    
    test_queries = [
        "How to treat diabetes?",
        "What are the symptoms of heart disease?",
        "Is cancer hereditary?",
        "What causes high blood pressure?",
        "How is Alzheimer's diagnosed?"
    ]
    
    print(f"📋 {len(test_queries)} requêtes de test\n")
    
    all_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*90}")
        print(f"# TEST {i}/{len(test_queries)}")
        print(f"{'#'*90}\n")
        
        results = comparer.compare_all(query, top_k=3)
        all_results.append((query, results))
        
        if i < len(test_queries):
            input("⏸️  Enter pour continuer...")
    
    # Résumé final
    print("\n" + "="*90)
    print("📊 RÉSUMÉ FINAL")
    print("="*90)
    
    total_t1 = sum(r['keyword'][1] for _, r in all_results)
    total_t2 = sum(r['semantic1'][1] for _, r in all_results)
    total_t3 = sum(r['semantic2'][1] for _, r in all_results)
    
    print(f"\n⏱️  Temps total ({len(test_queries)} requêtes):")
    print(f"   Classique:    {total_t1*1000:.2f}ms (avg: {total_t1*1000/len(test_queries):.2f}ms)")
    print(f"   Sémantique 1: {total_t2*1000:.2f}ms (avg: {total_t2*1000/len(test_queries):.2f}ms)")
    print(f"   Sémantique 2: {total_t3*1000:.2f}ms (avg: {total_t3*1000/len(test_queries):.2f}ms)")
    
    print("\n💡 Conclusions:")
    print("   ✅ Recherche classique: Rapide mais limitée (ne trouve pas toujours)")
    print("   ✅ Sémantique générale: Bonne, mais peut manquer de précision médicale")
    print("   ✅ Sémantique médicale: Meilleure précision pour questions médicales")
    
    print("\n" + "="*90 + "\n")
    
    comparer.close()
    print("✅ Test terminé!")


if __name__ == "__main__":
    main()