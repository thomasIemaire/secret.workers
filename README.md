python3.11 -m venv venv

venv\Scripts\activate ou source venv/bin/activate

pip install -r requirements.txt

ou

mkdir -p /data/tmp /data/pip-cache

TMPDIR=/data/tmp PIP_CACHE_DIR=/data/pip-cache XDG_CACHE_HOME=/data/.cache \
pip install -r requirements.txt

## Limitations connues

- L'entraînement NER repose sur une classification de tokens IOB : les entités qui se
  chevauchent ou sont imbriquées dans le texte ne peuvent donc pas être encodées.
  Le prétraitement déclenche désormais une erreur explicite si de telles entités sont
  détectées dans le jeu d'entraînement.

## Tester un modèle entraîné

1. **Activer l'environnement** et installer les dépendances comme indiqué ci-dessus.
2. **Configurer le chemin du modèle** dans `test_model.py` en décommentant la ligne
   correspondante (ou en remplaçant la valeur de `model_path`) vers le dossier qui
   contient `config.json` et `model.safetensors`/`pytorch_model.bin`.
3. **Fournir un texte d'entrée** : ajuster `raw_text` dans `test_model.py` avec un
   exemple contenant les entités que vous souhaitez reconnaître.
4. **Lancer l'inférence** :

   ```bash
   python test_model.py
   ```

Le script charge le pipeline `token-classification`, applique un léger nettoyage
du texte (espaces autour des slashs, suppression des espaces multiples) puis affiche
les entités détectées avec leurs labels, scores et texte reconstruit à partir des
positions de début/fin.
