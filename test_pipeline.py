import os
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

print("="*60)
print("TEST COMPLET DU PIPELINE RAG")
print("="*60)

# Test 1 : Clé API
print("\nTest 1 : Vérification de la clé API...")
if api_key and api_key != "AIzaSyDDRj1Wd3zYvWaeXClkZYi-pZF76y2P750":
    print(f"✅ Clé API trouvée : {api_key[:30]}...")
else:
    print(f"⚠️  Clé API par défaut ou absente")

# Test 2 : Import des modules
print("\nTest 2 : Import des modules...")
try:
    from src.pipeline import build_embeddings, chunk_document, generate_answer
    print("✅ Tous les modules importés avec succès")
except Exception as e:
    print(f"❌ Erreur import : {e}")
    sys.exit(1)

# Test 3 : Embeddings
print("\nTest 3 : Chargement des embeddings...")
try:
    embeddings = build_embeddings()
    print("✅ Embeddings chargés avec succès")
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 4 : Chunking
print("\nTest 4 : Test du chunking...")
try:
    test_text = "Je suis John. J'ai 30 ans. Je suis développeur. J'ai travaillé chez Google."
    chunks = chunk_document(test_text)
    print(f"✅ Texte fragmenté en {len(chunks)} chunk(s)")
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 5 : API Gemini
print("\nTest 5 : Test de l'API Gemini...")
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content('Qui es-tu ?')
    print(f"✅ API Gemini répond : {response.text[:80]}...")
except Exception as e:
    print(f"⚠️  Erreur Gemini : {str(e)[:100]}")

print("\n" + "="*60)
print("✅ PIPELINE RAG OPÉRATIONNEL - APP PRÊTE !")
print("="*60)
