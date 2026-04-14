
"""
CvRag - RAG Chatbot pour Analyse de CV
Filiere 4IASDR EMSI - Othmane Boushabi et Aya Khama
"""

import streamlit as st
from src.pipeline import reload_vector_store, answer_cv_only
from src.utils import load_cv_history, save_cv_history, export_conversation_to_pdf
import os
from datetime import datetime

st.set_page_config(
    page_title="CvRag - Analyse de CV avec IA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personnalisé
st.markdown("""<style>
[data-testid="stSidebar"] {background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);}
.main {background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);}
.main-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0.5rem;
}
.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}
.card:hover {
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.12);
    border-color: #667eea;
}
.stButton > button {
    width: 100%;
    border-radius: 8px;
    padding: 12px 24px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    font-weight: 600;
}
.response-box {
    background-color: #f8f9fa !important;
    border-left: 4px solid #667eea !important;
    padding: 12px !important;
    border-radius: 8px !important;
    color: #000000 !important;
}
.user-msg {background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%); padding: 12px; border-radius: 8px; margin: 8px 0;}
.asst-msg {background: linear-gradient(135deg, #f3f4f6 0%, #f0fdf4 100%); padding: 12px; border-radius: 8px; margin: 8px 0;}
.info-box {background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 8px;}
.success-box {background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); border-left: 4px solid #ec4899; padding: 12px; border-radius: 8px;}
</style>""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_cv_path" not in st.session_state:
    st.session_state.current_cv_path = None
if "cv_raw_text" not in st.session_state:
    st.session_state.cv_raw_text = ""
if "current_cv_name" not in st.session_state:
    st.session_state.current_cv_name = ""
if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<h1 style='color:#667eea; text-align:center'>📚 CvRag</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; text-align:center'>Analyse intelligente de CV</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📂 Gestion des CVs")
    st.markdown("**Telecharger un nouveau CV:**")
    
    uploaded_file = st.file_uploader(
        "Choisir PDF/DOCX",
        type=["pdf", "docx"],
        key="cv_uploader"
    )
    
    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner("Traitement du CV..."):
                result = reload_vector_store(file_path)
            
            if result['success']:
                st.session_state.vector_store = result['vector_store']
                st.session_state.current_cv_path = file_path
                st.session_state.current_cv_name = uploaded_file.name
                st.session_state.messages = []
                save_cv_history(uploaded_file.name, file_path)
                st.success(f"CV charge! ({result['chunks_count']} chunks)")
            else:
                st.error(f"Erreur: {result['message']}")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    
    st.markdown("---")
    st.markdown("**Historique des CVs:**")
    
    cv_history = load_cv_history()
    if cv_history:
        selected_cv_idx = st.selectbox(
            "Selectionner:",
            range(len(cv_history)),
            format_func=lambda i: f"{cv_history[i]['name']} ({cv_history[i]['date']})",
            key="cv_history_select"
        )
        
        if st.button("Charger"):
            selected_cv = cv_history[selected_cv_idx]
            try:
                with st.spinner("Chargement..."):
                    result = reload_vector_store(selected_cv['path'])
                
                if result['success']:
                    st.session_state.vector_store = result['vector_store']
                    st.session_state.current_cv_path = selected_cv['path']
                    st.session_state.current_cv_name = selected_cv['name']
                    st.session_state.messages = []
                    st.success("CV charge!")
                else:
                    st.error(f"Erreur: {result['message']}")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    else:
        st.info("Aucun historique")
    
    st.markdown("---")
    
    if st.session_state.current_cv_name:
        st.markdown(f"**CV Actuel:** {st.session_state.current_cv_name}")
        
        if st.button("Reinitialiser conversation"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Exporter en PDF"):
            if st.session_state.messages:
                try:
                    output_path = f"exports/{st.session_state.current_cv_name.split('.')[0]}_conversation.pdf"
                    os.makedirs("exports", exist_ok=True)
                    export_conversation_to_pdf(
                        st.session_state.messages,
                        st.session_state.current_cv_name,
                        st.session_state.cv_raw_text,
                        output_path
                    )
                    with open(output_path, "rb") as pdf_file:
                        st.download_button(
                            "Telecharger PDF",
                            pdf_file,
                            os.path.basename(output_path),
                            mime="application/pdf"
                        )
                    st.success("PDF genere!")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
            else:
                st.warning("Aucune conversation")
    else:
        st.info("Telecharger un CV d'abord")
    
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#64748b; font-size:0.9em'>CvRag<br/>4IASDR EMSI<br/>Othmane Boushabi et Aya Khama</p>", unsafe_allow_html=True)


# ==================== CONTENU PRINCIPAL ====================
st.markdown('<h1 class="main-title">CvRag</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b'>Analyse intelligente de CV avec IA</p>", unsafe_allow_html=True)
st.markdown("---")

if not st.session_state.vector_store:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""<div class='card' style='text-align:center'>
            <h2 style='color:#667eea'>Bienvenue dans CvRag</h2>
            <p style='color:#64748b'>Analysez vos CVs avec l'IA</p>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("### Questions suggerees:")
    
    questions = [
        "Quelle est la formation academique?",
        "Quelle est l'experience professionnelle?",
        "Quelles sont les competences techniques?",
        "Quels sont les projets realises?"
    ]
    
    cols = st.columns(len(questions))
    for i, (col, q) in enumerate(zip(cols, questions)):
        with col:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Analyse en cours..."):
                    try:
                        answer = answer_cv_only(q, st.session_state.vector_store)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("### Conversation")
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 12px; border-radius: 8px; border-left: 4px solid #2196F3; margin: 8px 0; color: #000000;">
            <strong>👤 Vous:</strong><br/>
            <div style="margin-top: 6px; font-size: 14px;">
            {msg['content']}
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 12px; border-radius: 8px; border-left: 4px solid #667eea; margin: 8px 0;">
            <strong>🤖 CvRag:</strong><br/>
            <div style="color: #000000; margin-top: 8px; font-size: 14px; line-height: 1.6;">
            {msg['content']}
            </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([0.88, 0.12])
    
    with col1:
        user_input = st.text_input(
            "Pose une question:",
            placeholder="Ex: Competences en Python?",
            key="user_input"
        )
    
    with col2:
        st.write("")  # Espacement vertical
        submit_btn = st.button("Envoyer", use_container_width=True, key="submit_btn")
    
    if submit_btn and user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.spinner("Analyse en cours..."):
            try:
                answer = answer_cv_only(user_input, st.session_state.vector_store)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                st.rerun()
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b; font-size:0.85em'>CvRag - 4IASDR EMSI | Othmane Boushabi et Aya Khama</p>", unsafe_allow_html=True)
