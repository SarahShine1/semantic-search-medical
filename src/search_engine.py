#!/usr/bin/env python3
"""
Moteur de Recherche Sémantique pour Questions Médicales
Phase 3 du TP: Implémentation du moteur de recherche
"""
import time
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from config import Config


class SemanticSearchEngine:
    """
    Moteur de recherche sémantique utilisant des embeddings vectoriels
    """
    
    def __init__(self):
        """Initialise le moteur de recherche"""
        print("🔧 Initialisation du moteur de recherche...")
        
        # Charger le modèle d'embeddings
        print(f"   📦 Chargement du modèle: {Config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Connexion à PostgreSQL
        print(f"   🔌 Connexion à PostgreSQL...")
        self.conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        
        print("   ✅ Moteur de recherche prêt!\n")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Convertit une requête en embedding vectoriel
        
        Args:
            query: Requête textuelle de l'utilisateur
            
        Returns:
            Vecteur numpy de dimension 384
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        # Normaliser pour cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> Tuple[List[Dict], float]:
        """
        Recherche sémantique basée sur les embeddings vectoriels
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de résultats à retourner
            category_filter: Filtrer par catégorie (optionnel)
            min_similarity: Seuil minimum de similarité
            
        Returns:
            (liste de résultats, temps d'exécution)
        """
        start_time = time.time()
        
        # 1. Encoder la requête
        query_embedding = self.encode_query(query)
        
        # 2. Préparer la requête SQL
        cursor = self.conn.cursor()
        
        # Use cosine distance operator from pgvector (<#>) and convert to similarity
        if category_filter:
            sql = """
                SELECT 
                    id,
                    question,
                    answer,
                    category,
                    qtype,
                    1 - (embedding <#> %s::vector) as similarity
                FROM medical_documents
                WHERE category = %s
                    AND (1 - (embedding <#> %s::vector)) >= %s
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
            """
            params = (
                query_embedding.tolist(),
                category_filter,
                query_embedding.tolist(),
                min_similarity,
                query_embedding.tolist(),
                top_k
            )
        else:
            sql = """
                SELECT 
                    id,
                    question,
                    answer,
                    category,
                    qtype,
                    1 - (embedding <#> %s::vector) as similarity
                FROM medical_documents
                WHERE (1 - (embedding <#> %s::vector)) >= %s
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
            """
            params = (
                query_embedding.tolist(),
                query_embedding.tolist(),
                min_similarity,
                query_embedding.tolist(),
                top_k
            )
        
        # 3. Exécuter la recherche
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        # 4. Formater les résultats
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'category': row[3],
                'qtype': row[4],
                'similarity': float(row[5]),
                'search_type': 'semantic'
            })
        
        cursor.close()
        
        search_time = time.time() - start_time
        return formatted_results, search_time
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> Tuple[List[Dict], float]:
        """
        Recherche traditionnelle par mots-clés (full-text search)
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de résultats
            category_filter: Filtrer par catégorie
            
        Returns:
            (liste de résultats, temps d'exécution)
        """
        start_time = time.time()
        cursor = self.conn.cursor()
        
        # Utiliser PostgreSQL full-text search
        if category_filter:
            sql = """
                SELECT 
                    id,
                    question,
                    answer,
                    category,
                    qtype,
                    ts_rank(to_tsvector('english', combined_text), 
                           plainto_tsquery('english', %s)) as rank
                FROM medical_documents
                WHERE category = %s
                    AND to_tsvector('english', combined_text) @@ 
                        plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s;
            """
            params = (query, category_filter, query, top_k)
        else:
            sql = """
                SELECT 
                    id,
                    question,
                    answer,
                    category,
                    qtype,
                    ts_rank(to_tsvector('english', combined_text), 
                           plainto_tsquery('english', %s)) as rank
                FROM medical_documents
                WHERE to_tsvector('english', combined_text) @@ 
                      plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s;
            """
            params = (query, query, top_k)
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'category': row[3],
                'qtype': row[4],
                'similarity': float(row[5]),  # rank score
                'search_type': 'keyword'
            })
        
        cursor.close()
        
        search_time = time.time() - start_time
        return formatted_results, search_time
    
    # Hybrid search removed per user request. Use semantic_search and keyword_search separately.
    
    def get_categories(self) -> List[str]:
        """Récupère la liste des catégories disponibles"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM medical_documents ORDER BY category;")
        categories = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return categories
    
    def get_statistics(self) -> Dict:
        """Récupère les statistiques de la base"""
        cursor = self.conn.cursor()
        
        # Nombre total de documents
        cursor.execute("SELECT COUNT(*) FROM medical_documents;")
        total_docs = cursor.fetchone()[0]
        
        # Distribution par catégorie
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM medical_documents 
            GROUP BY category 
            ORDER BY COUNT(*) DESC;
        """)
        category_dist = dict(cursor.fetchall())
        
        cursor.close()
        
        return {
            'total_documents': total_docs,
            'categories': category_dist
        }
    
    def close(self):
        """Ferme la connexion à la base de données"""
        self.conn.close()


def format_result(result: Dict, index: int, show_answer: bool = False) -> str:
    """
    Formate un résultat de recherche pour l'affichage
    
    Args:
        result: Dictionnaire avec les infos du document
        index: Numéro du résultat
        show_answer: Afficher la réponse complète ou juste un extrait
        
    Returns:
        Chaîne formatée
    """
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"🔍 Résultat #{index}")
    output.append(f"{'='*70}")
    output.append(f"📄 ID: {result['id']}")
    output.append(f"❓ Question: {result['question']}")
    output.append(f"🏷️  Catégorie: {result['category']}")
    output.append(f"📝 Type: {result['qtype']}")
    output.append(f"⭐ Score: {result['similarity']:.4f}")
    output.append(f"🔎 Méthode: {result['search_type']}")
    
    if show_answer:
        output.append(f"\n💬 Réponse:")
        output.append("-" * 70)
        # Limiter la réponse si trop longue
        answer = result['answer']
        if len(answer) > 500:
            output.append(answer[:500] + "...")
            output.append(f"\n[...réponse tronquée, {len(answer)} caractères total]")
        else:
            output.append(answer)
    else:
        # Montrer juste un extrait
        answer_preview = result['answer'][:200] + "..."
        output.append(f"\n💬 Extrait: {answer_preview}")
    
    return "\n".join(output)


if __name__ == "__main__":
    # Test rapide du moteur
    print("🧪 Test du moteur de recherche\n")
    
    engine = SemanticSearchEngine()
    
    # Exemple de recherche
    test_query = "How to treat high blood pressure?"
    print(f"🔍 Recherche: '{test_query}'\n")
    
    results, search_time = engine.semantic_search(test_query, top_k=3)
    
    print(f"⏱️  Temps de recherche: {search_time:.3f}s")
    print(f"📊 Résultats trouvés: {len(results)}")
    
    for idx, result in enumerate(results, 1):
        print(format_result(result, idx, show_answer=False))
    
    engine.close()
    print("\n✅ Test terminé!")