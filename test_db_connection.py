"""
Test de connexion PostgreSQL + pgvector
Vérifie que tout est correctement configuré
"""

import psycopg2
from config import Config

def test_database_connection():
    """
    Teste la connexion à PostgreSQL et vérifie pgvector
    """
    
    print("=" * 60)
    print("🔌 TEST DE CONNEXION À LA BASE DE DONNÉES")
    print("=" * 60)
    
    try:
        # 1. Connexion à PostgreSQL
        print("\n1️⃣ Connexion à PostgreSQL...")
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        print(f"   ✅ Connecté à {Config.DB_HOST}:{Config.DB_PORT}")
        
        cursor = conn.cursor()
        
        # 2. Vérifier la version PostgreSQL
        print("\n2️⃣ Vérification de PostgreSQL...")
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"   ✅ {version.split(',')[0]}")
        
        # 3. Vérifier pgvector
        print("\n3️⃣ Vérification de l'extension pgvector...")
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        if result:
            print(f"   ✅ pgvector version {result[0]} installée")
        else:
            print("   ❌ pgvector NON installée!")
            return False
        
        # 4. Vérifier la table medical_documents
        print("\n4️⃣ Vérification de la table 'medical_documents'...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'medical_documents'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("   ✅ Table 'medical_documents' existe")
            
            # Compter les documents
            cursor.execute("SELECT COUNT(*) FROM medical_documents;")
            count = cursor.fetchone()[0]
            print(f"   📊 Documents actuels: {count}")
            
            # Vérifier la structure
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'medical_documents'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            print("\n   📋 Structure de la table:")
            for col_name, col_type in columns:
                print(f"      - {col_name}: {col_type}")
        else:
            print("   ⚠️ Table 'medical_documents' n'existe pas encore")
            print("   💡 Pas de problème, elle sera créée automatiquement")
        
        # 5. Vérifier les index vectoriels
        print("\n5️⃣ Vérification des index...")
        cursor.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'medical_documents';
        """)
        indexes = cursor.fetchall()
        if indexes:
            print(f"   ✅ {len(indexes)} index trouvé(s)")
            for idx_name, idx_def in indexes:
                print(f"      - {idx_name}")
        else:
            print("   ⚠️ Aucun index (sera créé avec les données)")
        
        # 6. Test d'insertion simple
        print("\n6️⃣ Test d'insertion/suppression...")
        cursor.execute("""
            INSERT INTO medical_documents 
            (question, answer, category, source)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """, (
            "Test question",
            "Test answer",
            "test",
            "connection_test"
        ))
        test_id = cursor.fetchone()[0]
        print(f"   ✅ Insertion OK (ID: {test_id})")
        
        # Nettoyer le test
        cursor.execute("DELETE FROM medical_documents WHERE source = 'connection_test';")
        conn.commit()
        print("   ✅ Suppression OK")
        
        # 7. Résumé final
        print("\n" + "=" * 60)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("=" * 60)
        print("\n📌 Configuration actuelle:")
        print(f"   - Hôte: {Config.DB_HOST}")
        print(f"   - Port: {Config.DB_PORT}")
        print(f"   - Base de données: {Config.DB_NAME}")
        print(f"   - Utilisateur: {Config.DB_USER}")
        print(f"   - pgvector: Installé et fonctionnel")
        print(f"   - Table: Prête pour l'insertion")
        print("\n🎉 Votre base de données est prête!")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print("\n❌ ERREUR DE CONNEXION")
        print(f"   Détails: {e}")
        print("\n🔧 Solutions:")
        print("   1. Vérifier que Docker est lancé: docker ps")
        print("   2. Vérifier le container: docker-compose up -d")
        print("   3. Vérifier le .env (DB_HOST, DB_PORT, etc.)")
        return False
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database_connection()
    if not success:
        print("\n⚠️ Corrigez les erreurs avant de continuer!")
        exit(1)