"""
Prétraitement des données MedQuAD
- Nettoyage des textes
- Combinaison Question + Answer
- Préparation pour les embeddings
"""

import pandas as pd
import re
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def clean_text(text):
    """
    Nettoie un texte médical
    
    Args:
        text (str): Texte brut
        
    Returns:
        str: Texte nettoyé
    """
    if pd.isna(text):
        return ""
    
    # Convertir en string
    text = str(text)
    
    # Remplacer les sauts de ligne multiples par un seul espace
    text = re.sub(r'\n+', ' ', text)
    
    # Remplacer les espaces multiples par un seul
    text = re.sub(r'\s+', ' ', text)
    
    # Enlever les espaces au début et à la fin
    text = text.strip()
    
    return text

def combine_question_answer(row):
    """
    Combine Question + Answer pour avoir plus de contexte
    
    Args:
        row: Ligne du DataFrame
        
    Returns:
        str: Texte combiné
    """
    question = clean_text(row['Question'])
    answer = clean_text(row['Answer'])
    
    # Format: "Question: ... Answer: ..."
    combined = f"Question: {question} Answer: {answer}"
    
    return combined

def preprocess_dataset(input_path, output_path, sample_size=None):
    """
    Prétraite le dataset MedQuAD complet
    
    Args:
        input_path (str): Chemin du CSV brut
        output_path (str): Chemin du CSV traité
        sample_size (int, optional): Prendre seulement N documents (pour test)
    
    Returns:
        pd.DataFrame: Dataset prétraité
    """
    
    print("=" * 60)
    print("🧹 PRÉTRAITEMENT DES DONNÉES MEDQUAD")
    print("=" * 60)
    
    # 1. Charger les données
    print(f"\n1️⃣ Chargement du dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   ✅ {len(df)} documents chargés")
    print(f"   📋 Colonnes: {list(df.columns)}")
    
    # 2. Échantillonnage (optionnel - pour tester rapidement)
    if sample_size and sample_size < len(df):
        print(f"\n📊 Échantillonnage: {sample_size} documents (pour test rapide)")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"   ✅ {len(df)} documents sélectionnés")
    
    # 3. Vérifier les valeurs manquantes
    print("\n2️⃣ Vérification des valeurs manquantes...")
    missing = df[['Question', 'Answer']].isna().sum()
    print(f"   - Questions manquantes: {missing['Question']}")
    print(f"   - Réponses manquantes: {missing['Answer']}")
    
    # Supprimer les lignes avec Question OU Answer vide
    initial_len = len(df)
    df = df.dropna(subset=['Question', 'Answer'])
    removed = initial_len - len(df)
    if removed > 0:
        print(f"   🗑️ {removed} lignes supprimées (données manquantes)")
    else:
        print(f"   ✅ Aucune donnée manquante")
    
    # 4. Nettoyage des textes
    print("\n3️⃣ Nettoyage des textes...")
    df['question_clean'] = df['Question'].apply(clean_text)
    df['answer_clean'] = df['Answer'].apply(clean_text)
    print("   ✅ Textes nettoyés")
    
    # Vérifier la longueur des textes
    df['question_length'] = df['question_clean'].str.len()
    df['answer_length'] = df['answer_clean'].str.len()
    
    print(f"\n   📏 Statistiques de longueur:")
    print(f"      Questions:")
    print(f"         - Moyenne: {df['question_length'].mean():.0f} caractères")
    print(f"         - Min: {df['question_length'].min()}")
    print(f"         - Max: {df['question_length'].max()}")
    print(f"      Réponses:")
    print(f"         - Moyenne: {df['answer_length'].mean():.0f} caractères")
    print(f"         - Min: {df['answer_length'].min()}")
    print(f"         - Max: {df['answer_length'].max()}")
    
    # 5. Combiner Question + Answer
    print("\n4️⃣ Combinaison Question + Answer...")
    df['combined_text'] = df.apply(combine_question_answer, axis=1)
    df['combined_length'] = df['combined_text'].str.len()
    print("   ✅ Textes combinés")
    print(f"      - Longueur moyenne: {df['combined_length'].mean():.0f} caractères")
    
    # 6. Créer le DataFrame final
    print("\n5️⃣ Création du dataset final...")
    df_final = pd.DataFrame({
        'id': range(len(df)),
        'question': df['question_clean'],
        'answer': df['answer_clean'],
        'combined_text': df['combined_text'],
        'category': df['qtype'],
        'source': 'MedQuAD'
    })
    
    print(f"   ✅ {len(df_final)} documents prêts")
    
    # 7. Aperçu des données
    print("\n6️⃣ Aperçu des données prétraitées:")
    print("\n" + "=" * 60)
    for i in range(min(2, len(df_final))):
        row = df_final.iloc[i]
        print(f"\n📄 Document {i+1}:")
        print(f"   Catégorie: {row['category']}")
        print(f"   Question: {row['question'][:100]}...")
        print(f"   Answer: {row['answer'][:100]}...")
        print(f"   Combined: {row['combined_text'][:150]}...")
    print("\n" + "=" * 60)
    
    # 8. Sauvegarder
    print(f"\n7️⃣ Sauvegarde du dataset prétraité...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"   ✅ Sauvegardé: {output_path}")
    
    # 9. Statistiques finales
    print("\n" + "=" * 60)
    print("✅ PRÉTRAITEMENT TERMINÉ!")
    print("=" * 60)
    print(f"\n📊 Résumé:")
    print(f"   - Documents originaux: {initial_len}")
    print(f"   - Documents après nettoyage: {len(df_final)}")
    print(f"   - Catégories uniques: {df_final['category'].nunique()}")
    print(f"   - Fichier de sortie: {output_path}")
    print(f"   - Prêt pour la génération d'embeddings!")
    
    # Distribution par catégorie
    print(f"\n📈 Distribution par catégorie:")
    print(df_final['category'].value_counts().head(10))
    
    return df_final

if __name__ == "__main__":
    # Chemins
    input_csv = os.path.join(Config.RAW_DATA_DIR, 'medquad_raw.csv')
    output_csv = os.path.join(Config.PROCESSED_DATA_DIR, 'medquad_processed.csv')
    
    print(f"\n📂 Chemins:")
    print(f"   Input:  {input_csv}")
    print(f"   Output: {output_csv}")
    
    # Pour TEST RAPIDE: utilise sample_size=1000
    # Pour PRODUCTION: sample_size=None (tous les documents)
    
    # CHOISIS TON MODE:
    MODE = "PRODUCTION"  # ou "PRODUCTION"
    
    if MODE == "TEST":
        print("\n⚡ MODE TEST: 1000 documents seulement")
        df = preprocess_dataset(input_csv, output_csv, sample_size=1000)
    else:
        print("\n🏭 MODE PRODUCTION: Tous les documents")
        df = preprocess_dataset(input_csv, output_csv, sample_size=None)
    
    print("\n✨ Fichier prêt pour l'étape suivante: Génération d'embeddings")
