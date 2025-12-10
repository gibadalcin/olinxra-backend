# Olinx Plus Backend

<div align="center">

**API Backend para Plataforma de Realidade Aumentada com Reconhecimento Visual**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Digital Ocean](https://img.shields.io/badge/Digital%20Ocean-Hosted-0080FF.svg?logo=digitalocean&logoColor=white)](https://www.digitalocean.com/)

API REST de alta performance com reconhecimento visual de logos usando CLIP (ONNX) e FAISS

[Documentação](https://github.com/gibadalcin/olinxplus-docs) • [Admin UI](https://github.com/gibadalcin/olinxplus-adminui) • [Mobile App](https://github.com/gibadalcin/olinxplus)

</div>

---

## 📋 Visão Geral

O backend Olinx Plus é uma API FastAPI hospedada no Digital Ocean que fornece:

- 🔍 **Reconhecimento Visual Otimizado**: Busca de logos usando CLIP embeddings (ONNX) + pHash híbrido
- 🗄️ **Gestão de Conteúdo**: CRUD completo para conteúdos AR, logos e marcas
- 🎨 **Processamento de Mídia**: Upload, geração automática de GLBs e signed URLs
- 🔐 **Autenticação**: Firebase Authentication para AdminUI (app mobile é público)
- ☁️ **Cloud Storage**: Google Cloud Storage (buckets: olinxra-conteudo, olinxra-logos)
- 🤖 **IA**: CLIP ONNX (512d embeddings) + FAISS IVF para busca sublinear
- ⚡ **Performance**: Thresholds calibrados, crop inteligente, cache otimizado

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11 ou superior
- MongoDB (local ou Atlas)
- Conta Firebase com projeto configurado
- Google Cloud Storage bucket
- Git

### Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/gibadalcin/olinxplus-backend.git
cd olinxplus-backend
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as credenciais**

Coloque os seguintes arquivos no diretório raiz (não commitáveis):

**`firebase-cred.json`** - Firebase Admin SDK
```json
{
  "type": "service_account",
  "project_id": "seu-projeto",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "firebase-adminsdk@...",
  ...
}
```

**`cloud-storage-cred.json`** - Google Cloud Storage
```json
{
  "type": "service_account",
  "project_id": "seu-projeto-gcs",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  ...
}
```

5. **Configure variáveis de ambiente**

Crie arquivo `.env`:

```env
# MongoDB Atlas
MONGODB_URL=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/olinxplus

# Reconhecimento Visual (Thresholds Otimizados - Dez 2025)
SEARCH_COMBINED_THRESHOLD=0.50      # Combined CLIP+pHash threshold
SEARCH_MIN_MARGIN=0.01              # Mínimo entre top-1 e top-2
SEARCH_ACCEPTANCE_THRESHOLD=0.72    # Alta confiança
SEARCH_PHASH_WEIGHT=0.20            # Peso pHash (estrutural)
SEARCH_EMBEDDING_WEIGHT=0.80        # Peso CLIP (semântico)
SEARCH_CENTER_CROP_RATIO=1.0        # Crop desabilitado (app faz crop)
SEARCH_CROP_EXPAND_PCT=0            # Sem expansão

# Google Cloud Storage
GCS_BUCKET_CONTEUDO=olinxra-conteudo
GCS_BUCKET_LOGOS=olinxra-logos
```

6. **Execute o servidor**
```bash
# Desenvolvimento
python main.py
# ou
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Produção (Digital Ocean)
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

A API estará disponível em `http://localhost:8000`  
Documentação interativa: `http://localhost:8000/docs`

## 📡 Endpoints Principais

### Autenticação

```http
POST /token
POST /validate
```

### Logos

```http
GET    /logos                 # Listar todos os logos
GET    /logos/marca/{marca}   # Buscar logos por marca
POST   /logos                 # Upload de novo logo
POST   /logos/find-similar    # Busca por similaridade (CLIP)
DELETE /logos/{logo_id}       # Deletar logo
```

### Conteúdos AR

```http
GET    /conteudos                        # Listar conteúdos
GET    /conteudos/{marca}/{regiao}       # Buscar por marca/região
POST   /conteudos                        # Criar conteúdo
PUT    /conteudos/{marca}/{regiao}       # Atualizar conteúdo
DELETE /conteudos/{marca}/{regiao}       # Deletar conteúdo
```

### Imagens

```http
GET    /images                 # Listar imagens
POST   /images                 # Upload de imagem
DELETE /images/{filename}      # Deletar imagem
```

### Modelos 3D (GLB)

```http
POST   /upload-glb             # Upload de modelo GLB
GET    /carousel-glbs/{marca}/{regiao}  # Buscar GLBs de carousel
DELETE /glbs/{filename}        # Deletar GLB
```

### Debug (apenas desenvolvimento)

```http
GET    /debug/logos            # Verificar índice FAISS
GET    /debug/conteudos        # Listar todos os conteúdos
```

## 🏗️ Arquitetura

```
olinxplus-backend/
├── main.py                    # Entrypoint FastAPI (4146 linhas)
├── schemas.py                 # Modelos Pydantic
├── firebase_utils.py          # Firebase Admin + Auth
├── gcs_utils.py               # Google Cloud Storage
├── clip_utils.py              # CLIP embeddings (ONNX)
├── faiss_index.py             # Busca vetorial FAISS IVF
├── glb_generator.py           # Geração automática de GLB
├── requirements.txt           # Dependências Python
├── .env                       # Variáveis de ambiente (não commitável)
│
├── firebase-cred.json         # Firebase Admin SDK (não commitável)
├── cloud-storage-cred.json    # GCS credentials (não commitável)
│
├── clip_image_encoder.onnx    # Modelo CLIP (512d embeddings)
├── quantized_clip_model.onnx  # Modelo CLIP quantizado
├── faiss_index.index          # Índice FAISS (gerado dinamicamente)
├── logo_metadata.pkl          # Metadados dos logos indexados
│
├── docs/                      # Documentação técnica
│   └── CROP-OPTIMIZATION.md   # Otimizações de reconhecimento
│
├── tools/                     # Scripts utilitários
│   ├── add_topo_glb.py
│   ├── check_glbs_now.py
│   ├── generate_glbs_from_existing_images.py
│   ├── preprocess_variants.py
│   └── delete_anonymous_users.py
│
└── docs/                      # Documentação específica
```

## 🔧 Componentes Principais

### 1. Reconhecimento Visual (CLIP + FAISS)

O sistema usa CLIP para gerar embeddings visuais dos logos e FAISS para busca eficiente:

```python
# clip_utils.py
def clip_encode_image(image_data: bytes) -> np.ndarray:
    """Gera embedding de 512 dimensões usando CLIP"""
    
# faiss_index.py
def search_similar_logos(query_embedding: np.ndarray, top_k: int = 5):
    """Busca logos mais similares usando FAISS"""
```

**Fluxo de Reconhecimento:**
1. App envia imagem capturada
2. Backend gera embedding com CLIP
3. FAISS busca top-K logos mais similares
4. Retorna logos com scores de similaridade
5. App exibe conteúdo AR correspondente

### 2. Gestão de Conteúdo

Conteúdos AR são estruturados em blocos:

```python
{
  "marca": "oficina-g3",
  "regiao": "caxias-do-sul",
  "blocos": [
    {
      "tipo": "Imagem topo 1",
      "url": "gs://bucket/imagem.png",
      "signed_url": "https://storage.googleapis.com/..."
    },
    {
      "tipo": "Carousel 1",
      "items": [
        {
          "url": "gs://bucket/card1.png",
          "action": {
            "type": "external_link",
            "href": "https://example.com"
          }
        }
      ]
    },
    {
      "tipo": "modelo_3d",
      "url": "gs://bucket/modelo.glb"
    }
  ],
  "radius_m": 1000
}
```

### 3. Armazenamento em Nuvem

**Google Cloud Storage** para arquivos de mídia:
- URLs assinadas com expiração de 1 hora
- Organização por `user_id/arquivo.ext`
- CORS configurado para acesso do app

**MongoDB** para dados estruturados:
- Coleção `logos`: Metadados + embeddings
- Coleção `conteudos`: Blocos de conteúdo AR
- Coleção `carousel_glbs`: Modelos 3D

## 🔐 Segurança

### Autenticação

O backend suporta dois métodos de autenticação:

1. **Firebase ID Token** (recomendado)
```http
Authorization: Bearer <firebase_id_token>
```

2. **JWT Customizado** (legado)
```http
Authorization: Bearer <jwt_token>
```

### Variáveis de Ambiente Sensíveis

**NUNCA** commite:
- `.env` - Variáveis de ambiente
- `firebase-cred.json` - Credenciais Firebase
- `cloud-storage-cred.json` - Credenciais GCS
- `*.pkl` - Arquivos de metadados

Todos estão no `.gitignore` por segurança.

### CORS

Configure CORS no arquivo `gcs-cors.json` para permitir acesso do frontend:

```json
[
  {
    "origin": ["http://localhost:5173", "https://seu-dominio.com"],
    "method": ["GET", "POST", "PUT", "DELETE"],
    "maxAgeSeconds": 3600
  }
]
```

Aplique a configuração:
```bash
gsutil cors set gcs-cors.json gs://seu-bucket
```

## 🧪 Testes

### Testar Endpoints

**Listar logos:**
```bash
curl http://localhost:8000/logos
```

**Upload de logo:**
```bash
curl -X POST http://localhost:8000/logos \
  -H "Authorization: Bearer <token>" \
  -F "marca=nike" \
  -F "file=@logo.png"
```

**Buscar por similaridade:**
```bash
curl -X POST http://localhost:8000/logos/find-similar \
  -H "Content-Type: application/json" \
  -d '{"image_data": "<base64_encoded_image>"}'
```

**Verificar índice FAISS:**
```bash
curl http://localhost:8000/debug/logos
```

## 📊 Performance

### Otimizações Implementadas

- ✅ **ONNX Runtime**: CLIP inference 3-5x mais rápido que PyTorch
- ✅ **Modelo Quantizado**: 75% redução de tamanho sem perda de precisão
- ✅ **FAISS IVF**: Busca sublinear em milhões de vetores
- ✅ **Async I/O**: Motor para MongoDB async
- ✅ **Connection Pooling**: Reutilização de conexões HTTP/DB
- ✅ **Caching**: Embeddings armazenados no MongoDB

### Benchmarks

```
CLIP Encoding:    ~100ms por imagem (CPU)
FAISS Search:     ~5ms para 10K logos
API Latency:      ~150ms (find-similar endpoint)
Throughput:       ~50 req/s (single worker)
```

## 🐛 Troubleshooting

### Problema: "No module named 'onnxruntime'"
```bash
pip install onnxruntime>=1.17.0
```

### Problema: "MongoDB connection failed"
Verifique:
- MongoDB está rodando: `mongod --version`
- `MONGO_URI` está correta no `.env`
- Firewall permite conexão na porta 27017

### Problema: "GCS 403 Forbidden"
- Verifique `cloud-storage-cred.json`
- Confirme permissões do service account
- Execute `gcloud auth application-default login`

### Problema: "FAISS index not found"
```bash
# O índice é criado automaticamente no primeiro upload de logo
# Ou regenere manualmente:
python faiss_index.py
```

## 📈 Deploy

### DigitalOcean App Platform

1. Conecte o repositório GitHub
2. Configure variáveis de ambiente
3. Adicione credenciais como secrets
4. Deploy automático via push

### Docker (alternativa)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t olinxplus-backend .
docker run -p 8000:8000 --env-file .env olinxplus-backend
```

### Systemd (VM/VPS)

```ini
[Unit]
Description=Olinx Plus Backend API
After=network.target

[Service]
User=olinxplus
WorkingDirectory=/home/olinxplus/olinxplus-backend
Environment="PATH=/home/olinxplus/venv/bin"
ExecStart=/home/olinxplus/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📚 Recursos Adicionais

- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [CLIP Paper (OpenAI)](https://arxiv.org/abs/2103.00020)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup)
- [Google Cloud Storage](https://cloud.google.com/storage/docs)

## 🤝 Contribuindo

Ao contribuir para o backend:

1. Mantenha `requirements.txt` atualizado
2. Documente novos endpoints em docstrings
3. Adicione type hints em todas as funções
4. Siga PEP 8 para estilo de código
5. Teste endpoints antes de fazer PR

## 📄 Licença

Este projeto está sob a licença MIT.

---

<div align="center">
<strong>Olinx Plus Backend</strong> | Construído com FastAPI e ❤️
</div>
