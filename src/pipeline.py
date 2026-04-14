"""
Pipeline RAG pour l'analyse de CV
Projet : Chatbot RAG CV - 4IASDR
Encadrant : Pr. Idrissi Zouggari Nadia
"""

import os
import re
import pypdf
from docx import Document
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration de l'API Gemini
# Priorité : variable d'environnement > .env > clé codée en dur (fallback)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "AIzaSyDDRj1Wd3zYvWaeXClkZYi-pZF76y2P750":
    # Utiliser la clé fournie en dur comme fallback
    GOOGLE_API_KEY = "AIzaSyDDRj1Wd3zYvWaeXClkZYi-pZF76y2P750"
    print("ATTENTION: Clé API Gemini codée en dur. Pour la sécurité, utilisez une variable d'environnement.")

genai.configure(api_key=GOOGLE_API_KEY)


def load_cv(file_path):
    """
    Charge le contenu d'un CV depuis un fichier PDF ou DOCX
    
    Args:
        file_path (str): Chemin vers le fichier CV
        
    Returns:
        str: Contenu textuel du CV
    """
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""
    
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Nettoyer le texte encodé
                    if isinstance(page_text, bytes):
                        page_text = page_text.decode('utf-8', errors='replace')
                    text += page_text + "\n"
                    
    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    
    # Assurer l'encodage UTF-8 final
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    return text


def clean_text(text):
    """
    Nettoie le texte extrait (supprime espaces multiples, etc.)
    
    Args:
        text (str): Texte brut
        
    Returns:
        str: Texte nettoyé
    """
    # Assurer que le texte est correctement encodé en UTF-8
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Supprime les espaces multiples et les sauts de ligne excessifs
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_document(text, chunk_size=800, overlap=100):
    """
    Découpe le document en chunks pour le RAG
    
    Args:
        text (str): Texte à découper
        chunk_size (int): Taille des chunks
        overlap (int): Chevauchement entre chunks
        
    Returns:
        list: Liste de chunks de texte
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


def build_embeddings():
    """
    Construit le modèle d'embeddings Sentence Transformers
    
    Returns:
        HuggingFaceEmbeddings: Modèle d'embeddings
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def create_vector_store(chunks, embeddings):
    """
    Crée un vector store FAISS à partir des chunks
    
    Args:
        chunks (list): Liste de chunks de texte
        embeddings: Modèle d'embeddings
        
    Returns:
        FAISS: Vector store
    """
    return FAISS.from_texts(chunks, embeddings)


def retrieve_relevant_chunks(question, store, k=4):
    """
    Récupère les chunks les plus pertinents pour une question
    
    Args:
        question (str): Question posée
        store: Vector store FAISS
        k (int): Nombre de chunks à récupérer
        
    Returns:
        list: Liste de documents pertinents
    """
    return store.similarity_search(question, k=k)


def generate_answer(prompt):
    """
    Génère une réponse avec l'API Gemini
    Utilise les modèles disponibles avec fallbacks automatiques
    
    Args:
        prompt (str): Prompt à envoyer au modèle
        
    Returns:
        str: Réponse générée ou message d'erreur explicite
        
    Notes:
        - Essai d'abord avec gemini-2.5-flash (plus rapide et performant)
        - Fallback sur gemini-2.5-pro si 2.5-flash non disponible
        - Fallback sur gemini-2.0-flash si autres non disponibles
        - Gère les erreurs d'API quota et d'authentification
    """
    # Liste des modèles à essayer par ordre de préférence
    models_to_try = [
        'gemini-2.5-flash',      # Meilleur choix (2025)
        'gemini-2.5-pro',        # Fallback pour réponses plus détaillées
        'gemini-2.0-flash',      # Fallback (stable)
        'gemini-flash-latest'    # Alias disponible
    ]
    
    generation_config = {
        'temperature': 0.3,  # Déterministe pour des réponses précises
        'top_p': 0.8,
        'max_output_tokens': 500,
    }
    
    last_error = None
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            # Nettoyer et encoder correctement le texte
            answer_text = response.text
            # Faire un encode/decode pour nettoyer l'UTF-8
            if isinstance(answer_text, bytes):
                answer_text = answer_text.decode('utf-8', errors='replace')
            else:
                answer_text = answer_text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            return answer_text
            
        except Exception as e:
            last_error = str(e)
            # Essayer le modèle suivant
            continue
    
    # Si aucun modèle ne fonctionne, retourner un message d'erreur explicite
    if last_error:
        if "429" in last_error or "quota" in last_error.lower():
            return "Erreur : Quota API Gemini dépasse. Veuillez vérifier votre plan d'utilisation."
        elif "401" in last_error or "invalid" in last_error.lower():
            return "Erreur : Clé API Gemini invalide. Veuillez vérifier que GOOGLE_API_KEY est correcte."
        elif "404" in last_error or "not found" in last_error.lower():
            return "Erreur : Les modèles Gemini testés ne sont pas disponibles."
    
    return "Erreur : Impossible de contacter l'API Gemini. Verifiez votre connexion Internet."


def build_prompt(question, context):
    """
    Construit le prompt optimisé pour le modèle LLM
    Assure que le modèle répond uniquement sur la base du CV fourni
    
    Args:
        question (str): Question posée par l'utilisateur
        context (str): Contexte extrait du CV (chunks pertinents)
        
    Returns:
        str: Prompt formaté prêt à être envoyé au modèle
        
    Notes:
        - Enforce la politique "CV Only" pour éviter les hallucinations
        - Demande des réponses structurées et factuelles
        - Interdit l'invention ou la déduction d'informations
    """
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de CV pour la 4IASDR.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni ci-dessous.

RÈGLES STRICTES À RESPECTER :
1. Si l'information est présente dans le contexte, réponds de manière claire et structurée
2. Si l'information n'est PAS dans le contexte, réponds EXACTEMENT : "Cette information n'est pas présente dans le CV fourni"
3. INTERDIT : Ne déduis pas et n'invente pas d'informations
4. Reste factuel et précis - utilise les données du CV telles quelles
5. Pour les questions sur la formation, cite les diplômes avec les établissements et dates si disponibles
6. Pour l'expérience, cite les postes, entreprises et périodes mentionnés dans le CV

CONTEXTE DU CV :
{context}

QUESTION : {question}

RÉPONSE :"""
    return prompt


def reload_vector_store(file_path):
    """
    Recharge complètement la base vectorielle à partir d'un nouveau fichier CV
    Exécute l'ensemble du pipeline RAG (chargement -> nettoyage -> chunking -> embeddings -> indexation)
    
    Args:
        file_path (str): Chemin vers le fichier CV (PDF ou DOCX)
        
    Returns:
        dict: Dictionnaire contenant :
            - 'vector_store': L'objet FAISS créé
            - 'chunks_count': Nombre de chunks générés
            - 'success': Booléen indiquant le succès de l'opération
            - 'message': Message de statut
            
    Raises:
        Exception: En cas d'erreur lors de l'exécution du pipeline
        
    Notes:
        - Assure l'isolation des CVs : aucun mélange entre deux CVs
        - Exécute toutes les étapes en séquence
        - Idéal pour les uploads de nouveaux CVs
    """
    try:
        # Étape 1 : Chargement du CV
        text = load_cv(file_path)
        
        if not text:
            return {
                'success': False,
                'message': 'Erreur : Le CV est vide ou illisible',
                'vector_store': None,
                'chunks_count': 0
            }
        
        # Étape 2 : Nettoyage du texte
        cleaned_text = clean_text(text)
        
        # Étape 3 : Découpage en chunks
        chunks = chunk_document(cleaned_text)
        
        if not chunks:
            return {
                'success': False,
                'message': 'Erreur : Impossible de découper le document',
                'vector_store': None,
                'chunks_count': 0
            }
        
        # Étape 4 : Création des embeddings
        embeddings = build_embeddings()
        
        # Étape 5 : Création du vector store
        vector_store = create_vector_store(chunks, embeddings)
        
        return {
            'success': True,
            'message': f'Base vectorielle rechargée avec succès ({len(chunks)} chunks)',
            'vector_store': vector_store,
            'chunks_count': len(chunks)
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Erreur lors du rechargement : {str(e)}',
            'vector_store': None,
            'chunks_count': 0
        }


def answer_cv_only(question, store):
    """
    Répond à une question en se basant UNIQUEMENT sur le CV chargé
    Chaîne complète : récupération -> prompt -> génération
    
    Args:
        question (str): Question posée par l'utilisateur
        store: Vector store FAISS contenant le CV indexé
        
    Returns:
        str: Réponse générée par le modèle LLM
        
    Notes:
        - Applique la sécurité métier (CV Only)
        - Affiche le contexte récupéré pour le débogage
        - Fallback automatique sur d'autres modèles si erreur
    """
    # Étape 6 : Récupération des chunks pertinents
    relevant_docs = retrieve_relevant_chunks(question, store, k=4)
    context_parts = []
    for doc in relevant_docs:
        content = doc.page_content
        # Nettoyer l'encodage du contenu
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        context_parts.append(content)
    
    context = "\n".join(context_parts)
    
    # DEBUG : Affichage du contexte (utile pour le développement)
    print("\n" + "="*60)
    print("CONTEXTE RÉCUPÉRÉ POUR LA QUESTION")
    print("="*60)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("="*60 + "\n")
    
    # Étape 7 : Construction du prompt optimisé
    prompt = build_prompt(question, context)
    
    # Étape 8 : Génération de la réponse
    return generate_answer(prompt)
