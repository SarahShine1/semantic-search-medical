"""
Génère les embeddings pour les 2 modèles
"""
import pandas as pd
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, Model1Config, Model2Config


def generate_embeddings_for_model(model_config, input_csv):
    """
    Génère les embeddings pour UN modèle
    
    Args:
        model_config: Configuration du modèle (Model1Config ou Model2Config)
        input_csv: Chemin du CSV prétraité
    """
    print("\n" + "="*70)
    print(f"🧠 GÉNÉRATION EMBEDDINGS: {model_config.NAME}")
    print("="*70)
    
    # 1. Charger les données
    print(f"\n1️⃣ Chargement du dataset...")
    df = pd.read_csv(input_csv)
    texts = df['combined_text'].tolist()
    print(f"   ✅ {len(texts)} textes chargés")
    
    # 2. Charger le modèle
    print(f"\n2️⃣ Chargement du modèle '{model_config.NAME}'...")
    print(f"   ⏳ Cela peut prendre 1-2 minutes...")
    
    start = time.time()
    model = SentenceTransformer(model_config.NAME)
    load_time = time.time() - start
    
    print(f"   ✅ Modèle chargé en {load_time:.1f}s")
    print(f"   📏 Dimensions: {model_config.DIMENSIONS}")
    
    # 3. Générer embeddings
    print(f"\n3️⃣ Génération des embeddings...")
    print(f"   ⏳ Estimation: ~{len(texts) // 32 * 0.5:.0f}s")
    
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    gen_time = time.time() - start
    
    print(f"\n   ✅ Embeddings générés en {gen_time:.1f}s")
    print(f"   ⚡ Vitesse: {len(texts) / gen_time:.1f} docs/sec")
    
    # 4. Vérifier
    print(f"\n4️⃣ Vérification...")
    print(f"   - Shape: {embeddings.shape}")
    print(f"   - Attendu: ({len(texts)}, {model_config.DIMENSIONS})")
    
    assert embeddings.shape == (len(texts), model_config.DIMENSIONS), "Shape incorrecte!"
    assert not np.isnan(embeddings).any(), "NaN détectés!"
    
    print(f"   ✅ Vérification OK")
    
    # 5. Sauvegarder
    print(f"\n5️⃣ Sauvegarde...")
    os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
    np.save(model_config.EMBEDDINGS_FILE, embeddings)
    print(f"   ✅ Sauvegardé: {model_config.EMBEDDINGS_FILE}")
    
    print(f"\n✅ Terminé pour {model_config.NAME}!")
    
    return embeddings


def main():
    print("="*70)
    print("🔬 GÉNÉRATION EMBEDDINGS POUR 2 MODÈLES")
    print("="*70)
    
    input_csv = os.path.join(Config.PROCESSED_DATA_DIR, 'medquad_processed.csv')
    
    if not os.path.exists(input_csv):
        print(f"❌ Fichier non trouvé: {input_csv}")
        print("💡 Lance: python src/data_preprocessing.py")
        return
    
    print(f"\n📂 Input: {input_csv}")
    print(f"\n🎯 Modèles à générer:")
    print(f"   1. {Model1Config.NAME} → {Model1Config.DIMENSIONS}D")
    print(f"   2. {Model2Config.NAME} → {Model2Config.DIMENSIONS}D")
    
    # Demander confirmation
    print(f"\n⏱️ Temps estimé total: ~15-20 minutes")
    choice = input("\n▶️  Continuer? (y/n) > ").strip().lower()
    
    if choice != 'y':
        print("❌ Annulé")
        return
    
    # Modèle 1
    print("\n" + "#"*70)
    print("# MODÈLE 1/2")
    print("#"*70)
    embeddings1 = generate_embeddings_for_model(Model1Config, input_csv)
    
    # Modèle 2
    print("\n" + "#"*70)
    print("# MODÈLE 2/2")
    print("#"*70)
    embeddings2 = generate_embeddings_for_model(Model2Config, input_csv)
    
    # Résumé
    print("\n" + "="*70)
    print("🎉 TOUS LES EMBEDDINGS GÉNÉRÉS!")
    print("="*70)
    print(f"\n📊 Résumé:")
    print(f"   Modèle 1: {embeddings1.shape} → {Model1Config.EMBEDDINGS_FILE}")
    print(f"   Modèle 2: {embeddings2.shape} → {Model2Config.EMBEDDINGS_FILE}")
    print(f"\n✅ Prochaine étape: python insert_dual_models.py")


if __name__ == "__main__":
    main()