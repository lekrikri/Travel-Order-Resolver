# 🚀 API REST - Travel Order Resolver

API REST pour l'extraction automatique de villes de départ/arrivée à partir de commandes en langage naturel.

## 📋 Fonctionnalités

- ✅ Extraction de villes avec Qwen2.5-3B-Instruct
- ✅ Amélioration par ChromaDB RAG (93% accuracy)
- ✅ Support GPU (NVIDIA RTX 3060)
- ✅ Documentation interactive Swagger
- ✅ Health check et monitoring

## 🚀 Démarrage rapide

### 1. Lancer l'API

```bash
# Activer l'environnement
source venv_gpu/bin/activate

# Lancer l'API
python api.py
```

L'API sera disponible sur `http://localhost:8000`

### 2. Tester l'API

Dans un autre terminal :

```bash
# Activer l'environnement
source venv_gpu/bin/activate

# Lancer les tests
python test_api.py
```

## 📖 Documentation

### Endpoints disponibles

#### `GET /`
Page d'accueil avec documentation

#### `GET /health`
Health check de l'API

**Réponse:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "model_loaded": true
}
```

#### `GET /models`
Liste des modèles disponibles

**Réponse:**
```json
{
  "models": [
    {
      "name": "qwen2.5-only",
      "description": "Qwen2.5-3B-Instruct seul",
      "accuracy": "75%",
      "speed": "~6.5s/phrase"
    },
    {
      "name": "qwen2.5-rag",
      "description": "Qwen2.5-3B-Instruct + ChromaDB RAG",
      "accuracy": "93%",
      "speed": "~6.7s/phrase",
      "recommended": true
    }
  ],
  "chromadb_examples": 100
}
```

#### `POST /extract`
Extrait les villes de départ et d'arrivée

**Requête:**
```json
{
  "sentence": "Je voudrais un billet de train de Paris à Lyon",
  "use_rag": true
}
```

**Réponse:**
```json
{
  "sentence": "Je voudrais un billet de train de Paris à Lyon",
  "departure": "Paris",
  "arrival": "Lyon",
  "confidence": 0.95,
  "method": "Qwen2.5 + ChromaDB RAG"
}
```

## 💻 Exemples d'utilisation

### Avec curl

```bash
# Health check
curl http://localhost:8000/health

# Extraction avec RAG
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Besoin d un billet Marseille Bordeaux", "use_rag": true}'

# Extraction sans RAG
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Besoin d un billet Marseille Bordeaux", "use_rag": false}'
```

### Avec Python

```python
import requests

# Extraction
response = requests.post(
    "http://localhost:8000/extract",
    json={
        "sentence": "Un billet Lille Nice s'il vous plaît",
        "use_rag": True
    }
)

result = response.json()
print(f"Départ: {result['departure']}")
print(f"Arrivée: {result['arrival']}")
print(f"Confiance: {result['confidence']:.0%}")
```

### Avec JavaScript

```javascript
fetch('http://localhost:8000/extract', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    sentence: 'Je veux aller de Strasbourg à Toulouse',
    use_rag: true
  })
})
.then(response => response.json())
.then(data => {
  console.log('Départ:', data.departure);
  console.log('Arrivée:', data.arrival);
  console.log('Confiance:', data.confidence);
});
```

## 📊 Performances

| Méthode | Accuracy | Vitesse |
|---------|----------|---------|
| Qwen2.5 seul | 75% | ~6.5s/phrase |
| **Qwen2.5 + RAG** | **93%** | ~6.7s/phrase |

## 🔧 Configuration

### Variables d'environnement

Aucune variable d'environnement requise. L'API utilise les configurations par défaut.

### Ports

Par défaut, l'API écoute sur le port `8000`. Pour changer :

```python
# Dans api.py, dernière ligne
uvicorn.run(app, host="0.0.0.0", port=8080)  # Changer le port
```

## 📚 Documentation interactive

Une fois l'API lancée, accédez à :

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Page d'accueil:** http://localhost:8000/

## 🐛 Dépannage

### L'API ne démarre pas

```bash
# Vérifier que l'environnement est activé
source venv_gpu/bin/activate

# Vérifier que les dépendances sont installées
pip install fastapi uvicorn[standard]
```

### Erreur 503 "Modèle non chargé"

Le modèle peut prendre quelques secondes à charger au démarrage. Attendez que vous voyiez :

```
✅ API PRÊTE - MODÈLE CHARGÉ
```

### Problème de mémoire GPU

Si vous manquez de VRAM, l'API passera automatiquement en mode CPU (plus lent).

## 📦 Déploiement en production

### Avec Uvicorn (simple)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

### Avec Gunicorn + Uvicorn (production)

```bash
pip install gunicorn

gunicorn api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Note:** Utilisez `--workers 1` car le modèle est chargé en mémoire GPU.

### Avec Docker

```dockerfile
FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔐 Sécurité

Pour un déploiement en production :

1. **Authentification**: Ajoutez un système d'authentification (API keys, JWT)
2. **HTTPS**: Utilisez un reverse proxy (Nginx) avec SSL
3. **Rate limiting**: Limitez le nombre de requêtes par IP
4. **CORS**: Configurez les origines autorisées

## 📝 License

Projet académique - T-AIA-911-PAR_14
