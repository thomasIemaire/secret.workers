from transformers import pipeline
import re

# --- CONFIGURATION ---
# Remplace ceci par le chemin réel vers ton dossier modèle (là où il y a config.json et model.safetensors/pytorch_model.bin)

# model_path = "./sardine.agents/date-parts/2.16"
# model_path = "./sardine.agents/payslip-employee-id/4.1" 
# model_path = "./sardine.agents/invoice-number/1.2" 
# model_path = "./sardine.agents/postal-address/1.15" 
# model_path = "./sardine.agents/order-number/1.0" 
model_path = "./sardine.agents/customer-number/1.1" 


# Exemple de texte difficile (avec les slashs collés)
# raw_text = "Référence de la facture :  FR71123803 Type de facture :  Renouvellement  Date d'émission :  01/07/2025  Commande :  BC230780805  Identifiant Client :  lt472563-ovh "
raw_text = "Thomas Lemaire 464 Boulevard des Tamaris 12850 Onet le chateau France Numéro client : 44436142482245 Numéro de commande : ORD-1234-123456 Numéro de document : 203118796228 Facture du 22 oct. 2024 "
# raw_text = "Thomas Lemaire 464 Boulevard des Tamaris 12850 Onet le chateau France"

try:
    # --- 1. CHARGEMENT DU MODÈLE ---
    print(f"Chargement du modèle depuis {model_path}...")
    # aggregation_strategy="simple" est très utile : il recombine automatiquement les B-TAG et I-TAG
    # pour te donner une seule entité complète (ex: "1925" au lieu de "1", "9", "25").
    nlp = pipeline(
        "token-classification", 
        model=model_path, 
        tokenizer=model_path, 
        aggregation_strategy="simple" 
    )

    # --- 2. PRÉ-TRAITEMENT (Le correctif 'Builder') ---
    # On ajoute des espaces autour des slashs pour aider le tokenizer Camembert
    text_cleaned = re.sub(r"([/])", r" \1 ", raw_text)
    # On nettoie les espaces multiples éventuels
    text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()
    
    print(f"\nTexte original : '{raw_text}'")
    print(f"Texte nettoyé  : '{text_cleaned}' (envoyé au modèle)\n")

    # --- 3. INFÉRENCE ---
    results = nlp(text_cleaned)

    # --- 4. AFFICHAGE DES RÉSULTATS ---
    if not results:
        print("Aucune entité détectée.")
    else:
        print("-" * 80)
        print(f"{'Entité (Reconstruite)':<20} | {'Entité (CORRIGÉE)':<20} | {'Label':<25} | {'Score'}")
        print("-" * 80)
        
        for entity in results:
            label = entity['entity_group']
            score = entity['score']
            
            # CE QUE L'IA ESSAIE DE RECOLLER (Souvent cassé)
            mot_reconstruit = entity['word'] 
            
            # --- LA SOLUTION MAGIQUE ---
            # On utilise les positions exactes (start/end) pour extraire 
            # le texte directement depuis la source propre.
            start = entity['start']
            end = entity['end']
            
            # On découpe dans text_cleaned car c'est celui que l'IA a analysé
            mot_original = text_cleaned[start:end]
            
            print(f"{mot_reconstruit:<20} | {mot_original:<20} | {label:<25} | {score:.4f}")
            
    print("-" * 80)

except Exception as e:
    print(f"\nErreur : Impossible de charger le modèle.\nVérifie que le chemin '{model_path}' est correct.")
    print(f"Détail de l'erreur : {e}")