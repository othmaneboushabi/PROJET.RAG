"""
Package RAG pour l'analyse de CV
Projet TP 4IASDR - Chatbot RAG CV
Encadrante : Pr. Idrissi Zouggari Nadia

Exporte toutes les fonctions du pipeline RAG pour utilisation externe.
"""
from .pipeline import (
    load_cv,
    clean_text,
    chunk_document,
    build_embeddings,
    create_vector_store,
    retrieve_relevant_chunks,
    build_prompt,
    generate_answer,
    answer_cv_only,
    reload_vector_store
)

__all__ = [
    'load_cv',
    'clean_text',
    'chunk_document',
    'build_embeddings',
    'create_vector_store',
    'retrieve_relevant_chunks',
    'build_prompt',
    'generate_answer',
    'answer_cv_only',
    'reload_vector_store'
]

__version__ = "1.0.0"
__author__ = "Étudiant 4IASDR"
__description__ = "Pipeline RAG complet pour chatbot CV"
