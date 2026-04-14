# Configuration de l'API Google Gemini

## Problème: Le LLM ne répond pas

**Raison**: La clé API Gemini fournie n'est plus valide (exposée publiquement)

## Solution: Créer une nouvelle clé API

### Étapes:

1. **Allez sur** https://ai.google.dev/
2. **Cliquez sur** "Get API Key" ou allez sur https://aistudio.google.com/app/apikey
3. **Créez une nouvelle clé** (Create API Key)
4. **Copiez la clé**
5. **Ouvrez le fichier** `.env` dans le projet
6. **Remplacez** la clé existante:

```
GOOGLE_API_KEY=VOTRE_CLE_COPIEE_ICI
```

**Exemple:**
```
GOOGLE_API_KEY=AIzaSy...votre_vraie_cle...
```

7. **Sauvegardez** le fichier `.env`
8. **Redémarrez** Streamlit: `streamlit run app.py`

## Sécurité

⚠️ **IMPORTANT**: 
- Ne commitez JAMAIS le `.env` avec votre vraie clé
- Utilisez `.env.example` pour les templates
- Ajoutez `.env` à `.gitignore` (déjà fait ✓)

## Test rapide

Une fois la clé mise à jour:
1. Ouvrez l'app Streamlit
2. Tél
échargez un CV
3. Posez une question
4. Vous devriez voir une réponse du LLM

Si ça ne marche pas:
- Vérifiez votre quota API sur https://makersuite.google.com/app/apikey
- Vérifiez que le modèle `gemini-2.0-flash` ou `gemini-1.5-flash` est disponible
- Consultez les logs Streamlit pour les erreurs détaillées
