
import os
import re
import pypdf
from docx import Document
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration de l'API Gemini
GOOGLE_API_KEY = "AIzaSyCjuZV3HRIpK-blFD02Z5CB_akR9wiCKJ0"
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
                    text += page_text + "\n"
                    
    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    
    return text


def clean_text(text):
    """
    Nettoie le texte extrait (supprime espaces multiples, etc.)
    
    Args:
        text (str): Texte brut
        
    Returns:
        str: Texte nettoyé
    """
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
    CORRECTION : Utilise gemini-2.0-flash-exp au lieu de gemini-1.5-flash
    
    Args:
        prompt (str): Prompt à envoyer au modèle
        
    Returns:
        str: Réponse générée
    """
    try:
        # Essai avec le modèle le plus récent
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        generation_config = {
            'temperature': 0.3,  # Plus déterministe pour des réponses précises
            'top_p': 0.8,
            'max_output_tokens': 500,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
        
    except Exception as e:
        # Fallback sur gemini-pro si le modèle 2.0 n'est pas disponible
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"❌ Erreur lors de la génération : {str(e2)}"


def answer_cv_only(question, store):
    """
    Répond à une question en se basant uniquement sur le CV chargé
    
    Args:
        question (str): Question de l'utilisateur
        store: Vector store FAISS contenant le CV
        
    Returns:
        str: Réponse générée
    """
    # Récupération des chunks pertinents
    relevant_docs = retrieve_relevant_chunks(question, store)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # DEBUG : Affichage du contexte (utile pour le développement)
    print("\n" + "="*60)
    print("📄 CONTEXTE RÉCUPÉRÉ")
    print("="*60)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("="*60 + "\n")
    
    # Construction du prompt optimisé
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de CV. 
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni ci-dessous.

RÈGLES IMPORTANTES :
1. Si l'information est présente dans le contexte, réponds de manière claire et structurée
2. Si l'information n'est PAS dans le contexte, réponds : "Cette information n'est pas présente dans le CV fourni"
3. Ne déduis pas et n'invente pas d'informations
4. Reste factuel et précis
5. Pour les questions sur la formation, cite les diplômes avec les établissements et dates si disponibles

CONTEXTE DU CV :
{context}

QUESTION : {question}

RÉPONSE :"""
    
    return generate_answer(prompt)
