"""
Utilitaires pour CVio RAG Chatbot
- Gestion de l'historique des CVs
- Export PDF de conversations
"""

import json
import os
from datetime import datetime
from typing import List, Dict

# ==================== HISTORIQUE DES CVs ====================

CV_HISTORY_FILE = "data/.cv_history.json"


def load_cv_history() -> List[Dict]:
    """
    Charge l'historique des CVs depuis le fichier JSON.
    
    Returns:
        List[Dict]: Liste des CVs avec 'name', 'path', 'date'
    """
    if not os.path.exists(CV_HISTORY_FILE):
        return []
    
    try:
        with open(CV_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de l'historique: {e}")
        return []


def save_cv_history(cv_name: str, cv_path: str) -> bool:
    """
    Ajoute un CV à l'historique avec métadonnées.
    
    Args:
        cv_name (str): Nom du CV
        cv_path (str): Chemin du fichier CV
        
    Returns:
        bool: True si succès, False sinon
    """
    try:
        history = load_cv_history()
        
        # Vérifier si le CV existe déjà
        for cv in history:
            if cv['path'] == cv_path:
                # Mettre à jour la date
                cv['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        else:
            # Ajouter le nouveau CV
            history.append({
                'name': cv_name,
                'path': cv_path,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Garder seulement les 10 derniers CVs
        history = history[-10:]
        
        # Sauvegarder
        os.makedirs("data", exist_ok=True)
        with open(CV_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'historique: {e}")
        return False


# ==================== EXPORT PDF ====================

def export_conversation_to_pdf(messages: List[Dict], cv_name: str, cv_text: str, output_path: str) -> bool:
    """
    Exporte la conversation complète en PDF formaté.
    
    Args:
        messages (List[Dict]): Liste des messages {'role': 'user'/'assistant', 'content': '...'}
        cv_name (str): Nom du CV
        cv_text (str): Contenu du CV
        output_path (str): Chemin de sortie du PDF
        
    Returns:
        bool: True si succès, False sinon
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        
        # Créer le document PDF
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch,
            title="CVio - Rapport d'Analyse"
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Contenu du PDF
        content = []
        
        # Titre
        content.append(Paragraph("📊 CVio - Rapport d'Analyse de CV", title_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Métadonnées
        meta_table = [
            ['Nom du CV:', cv_name],
            ['Date d\'export:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Nombre de messages:', str(len(messages))],
        ]
        
        table = Table(meta_table, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        content.append(table)
        content.append(Spacer(1, 0.3*inch))
        
        # Conversation
        content.append(Paragraph("💬 Conversation", heading_style))
        content.append(Spacer(1, 0.1*inch))
        
        for msg in messages:
            if msg.get('role') == 'user':
                # Limiter la longueur
                msg_content = msg.get('content', '')[:500]
                content.append(Paragraph(
                    f"<b>👤 Question:</b><br/>{msg_content}",
                    styles['Normal']
                ))
            else:
                # Limiter la longueur
                msg_content = msg.get('content', '')[:500]
                content.append(Paragraph(
                    f"<b>🤖 CVio:</b><br/>{msg_content}",
                    styles['Normal']
                ))
            
            content.append(Spacer(1, 0.15*inch))
        
        # CV original
        if cv_text:
            content.append(PageBreak())
            content.append(Paragraph("📄 CV Original", heading_style))
            content.append(Spacer(1, 0.1*inch))
            
            # Limiter à 3000 caractères pour ne pas surcharger le PDF
            cv_preview = cv_text[:3000] + "..." if len(cv_text) > 3000 else cv_text
            content.append(Paragraph(
                cv_preview.replace('\n', '<br/>'),
                styles['Normal']
            ))
        
        # Footer
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph(
            "<i>Généré par CVio | RAG Chatbot pour Analyse de CV | 4IASDR - EMSI</i>",
            ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
        ))
        
        # Générer le PDF
        doc.build(content)
        return True
        
    except Exception as e:
        print(f"Erreur lors de la génération du PDF: {e}")
        return False
