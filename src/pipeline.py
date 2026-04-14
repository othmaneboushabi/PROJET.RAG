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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration de l'API Gemini
# Priorité : variable d'environnement > .env > clé codée en dur (fallback)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "AIzaSyCReIn2cbnHyZ6IxqocHFssHrX73Rpf0NA":
    # Utiliser la clé fournie en dur comme fallback
    GOOGLE_API_KEY = "AIzaSyCReIn2cbnHyZ6IxqocHFssHrX73Rpf0NA"
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
    Nettoie le texte extrait (supprime HTML, espaces, caractères spéciaux)
    
    Args:
        text (str): Texte brut pouvant contenir du HTML
        
    Returns:
        str: Texte nettoyé et lisible
    """
    import html
    
    # Assurer que le texte est correctement encodé en UTF-8
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # 1. Décode les entités HTML (&lt; -> <, &gt; -> >)
    text = html.unescape(text)
    
    # 2. Supprime les balises HTML/XML complètes
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Supprime les caractères de contrôle non imprimables
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    
    # 4. Supprime les espaces multiples et les sauts de ligne excessifs
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Supprime les astérisques multiples (**) qui ne servent à rien
    text = re.sub(r'\*{2,}', '', text)
    
    # 6. Nettoie les tirets et caractères de formatage
    text = re.sub(r'[-–—]+', '-', text)
    
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
    """
    # Liste des modèles à essayer par ordre de préférence
    models_to_try = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-2.0-flash',
        'gemini-flash-latest'
    ]
    
    generation_config = {
        'temperature': 0.5,  # Équilibre entre créativité et précision
        'top_p': 0.9,
        'max_output_tokens': 1000,  # Augmenté pour réponses détaillées
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
            if isinstance(answer_text, bytes):
                answer_text = answer_text.decode('utf-8', errors='replace')
            else:
                answer_text = answer_text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            return answer_text
            
        except Exception as e:
            last_error = str(e)
            continue
    
    # Si aucun modèle ne fonctionne
    if last_error:
        if "429" in last_error or "quota" in last_error.lower():
            return "Erreur : Quota API Gemini dépassé. Veuillez vérifier votre plan d'utilisation."
        elif "401" in last_error or "invalid" in last_error.lower():
            return "Erreur : Clé API Gemini invalide."
        elif "404" in last_error:
            return "Erreur : Les modèles Gemini testés ne sont pas disponibles."
    
    return "Erreur : Impossible de contacter l'API Gemini. Verifiez votre connexion Internet."


def build_prompt(question, context):
    """
    Construit un prompt optimisé pour générer des réponses claires et détaillées
    
    Args:
        question (str): Question posée par l'utilisateur
        context (str): Contexte extrait du CV
        
    Returns:
        str: Prompt formaté pour le modèle LLM
    """
    prompt = f"""Tu es un expert en analyse de CV. Lis attentivement le contexte fourni et réponds à la question de manière CLAIRE, DÉTAILLÉE et STRUCTURÉE.

DIRECTIVES :
1. Fournís une réponse COMPLÈTE avec tous les détails disponibles
2. Si l'information est présente, développe ta réponse avec les dates, entreprises, technologies mentionnées
3. Organise ta réponse avec des bullet points ou numérotation si nécessaire
4. Si l'information n'est PAS dans le contexte, dis simplement : "Cette information n'est pas présente dans le CV"
5. Ne dééduis pas - reste fidèle aux données du CV
6. Sois aussi spécifique et détaillé que possible

CONTEXTE DU CV :
{context}

QUESTION DE L'UTILISATEUR : {question}

RÉPONSE DÉTAILLÉE :"""
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
    Répond à une question en se basant sur le CV chargé
    Chaîne complète : récupération → prompt → génération
    
    Args:
        question (str): Question posée par l'utilisateur
        store: Vector store FAISS contenant le CV indexé
        
    Returns:
        str: Réponse générée par le modèle LLM
    """
    # Récupération des chunks pertinents (augmenté à 6 pour plus de contexte)
    relevant_docs = retrieve_relevant_chunks(question, store, k=6)
    context_parts = []
    for doc in relevant_docs:
        content = doc.page_content
        # Nettoyer l'encodage du contenu
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        context_parts.append(content)
    
    context = "\n".join(context_parts)
    
    # DEBUG : Affichage du contexte
    print("\n" + "="*60)
    print("CONTEXTE RÉCUPÉRÉ POUR LA QUESTION")
    print("="*60)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("="*60 + "\n")
    
    # Construction du prompt optimisé
    prompt = build_prompt(question, context)
    
    # Génération de la réponse
    return generate_answer(prompt)
