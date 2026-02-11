from datasets import load_dataset
import pandas as pd
import os
from config import Config

def download_medquad():
    """
    Télécharge le dataset MedQuAD depuis Hugging Face
    """
    
    print("📥 Téléchargement du dataset MedQuAD...")
    print("⏳ Cela peut prendre 2-3 minutes...\n")
    
    try:
        # Télécharger depuis Hugging Face
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        
        # Convertir en DataFrame pandas
        df = pd.DataFrame(dataset['train'])
        
        print(f"✅ Dataset téléchargé!")
        print(f"📊 Nombre de Q&A: {len(df)}")
        print(f"\n📋 Colonnes disponibles: {list(df.columns)}")
        print(f"\n🔍 Aperçu des premières lignes:\n")
        print(df.head(3))
        
        # Créer le dossier data/raw s'il n'existe pas
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        
        # Sauvegarder en CSV
        csv_path = os.path.join(Config.RAW_DATA_DIR, 'medquad_raw.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Dataset sauvegardé: {csv_path}")
        
        # Statistiques
        print("\n📈 Statistiques:")
        print(f"   - Questions uniques: {df['Question'].nunique()}")
        print(f"   - Types de questions (qtype): {df['qtype'].nunique()}")
        print(f"\n🏷️ Catégories de questions:")
        print(df['qtype'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return None

if __name__ == "__main__":
    df = download_medquad()
    
    if df is not None:
        print("\n✨ Téléchargement terminé avec succès!")
        print("📂 Fichier disponible dans: data/raw/medquad_raw.csv")