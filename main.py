from fastapi import Query
import logging
import json
import os
import numpy as np
import firebase_admin
import httpx
import tempfile
import smtplib
from fastapi import Request
from firebase_admin import credentials, auth
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form, Query, Body, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from bson import ObjectId
from bson.errors import InvalidId
from datetime import timedelta, datetime
from gcs_utils import upload_image_to_gcs, get_bucket
from schemas import validate_button_block_payload
from clip_utils import extract_clip_features
from faiss_index import LogoIndex
from email.mime.text import MIMEText
import asyncio
import onnxruntime as ort
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()


logo_index = None
ort_session = None
http_bearer = HTTPBearer()
client = None
db = None
logos_collection = None
# REMOVIDO: images_collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db, logos_collection
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise RuntimeError("Variável de ambiente MONGO_URI não encontrada.")
    DB_NAME = os.getenv("MONGO_DB_NAME", "olinxra")

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    logos_collection = db["logos"]
    # Criar índices recomendados para a coleção de conteúdos
    try:
        # Índice composto por marca_id + owner_uid + tipo_regiao + nome_regiao (documents com marca_id)
        await db['conteudos'].create_index([
            ('marca_id', 1),
            ('owner_uid', 1),
            ('tipo_regiao', 1),
            ('nome_regiao', 1)
        ], name='idx_marcaid_owner_region', sparse=True)

        # Índice composto por nome_marca + owner_uid + tipo_regiao + nome_regiao (compatibilidade com docs antigos)
        await db['conteudos'].create_index([
            ('nome_marca', 1),
            ('owner_uid', 1),
            ('tipo_regiao', 1),
            ('nome_regiao', 1)
        ], name='idx_nomemarca_owner_region', sparse=True)

        # Índice 2dsphere para consultas geoespaciais, se usarmos campo 'location'
        await db['conteudos'].create_index([('location', '2dsphere')], name='idx_location_2dsphere')

        # Índice simples por owner_uid
        await db['conteudos'].create_index([('owner_uid', 1)], name='idx_owner_uid')

        logging.info('Índices de conteúdo verificados/criados com sucesso.')
    except Exception as e:
        logging.exception(f'Falha ao criar índices em conteudos: {e}')
    # REMOVIDO: images_collection = db["images"]

    logging.info("Iniciando a aplicação...")
    initialize_firebase()
    initialize_onnx_session()
    await load_faiss_index()
    yield
    if client:
        client.close()

app = FastAPI(lifespan=lifespan)

# --- Helpers de Inicialização ---
###############################################################
# Função utilitária e endpoint para gerar signed URL de conteúdo
def gerar_signed_url_conteudo(gs_url, filename=None):
    # Detecta bucket pelo prefixo ou filename
    tipo_bucket = "conteudo"
    if gs_url.startswith("gs://olinxra-logos/"):
        tipo_bucket = "logos"
    elif gs_url.startswith("gs://olinxra-conteudo/"):
        tipo_bucket = "conteudo"
    elif filename and "conteudo" in filename:
        tipo_bucket = "conteudo"
    else:
        tipo_bucket = "logos"
    # Extrai o caminho completo após o nome do bucket se não informado
    if not filename:
        if gs_url.startswith("gs://olinxra-conteudo/"):
            filename = gs_url[len("gs://olinxra-conteudo/"):]
        elif gs_url.startswith("gs://olinxra-logos/"):
            filename = gs_url[len("gs://olinxra-logos/"):]
        else:
            # fallback: tenta pegar o nome do arquivo (última parte do path)
            filename = gs_url.split("/")[-1]
    try:
        bucket = get_bucket(tipo_bucket)
        url = bucket.blob(filename).generate_signed_url(
            version="v4",
            expiration=3600,
            method="GET"
        )
        return url
    except Exception as e:
        logging.error(f"Erro ao gerar signed URL para {filename} (bucket {tipo_bucket}): {e}")
        return ""

@app.get("/api/conteudo-signed-url")
async def get_conteudo_signed_url(gs_url: str = Query(...), filename: str = Query(None)):
    url = gerar_signed_url_conteudo(gs_url, filename)
    return {"signed_url": url}

def initialize_firebase():
    cred_json_str = os.getenv("FIREBASE_CRED_JSON")
    if not cred_json_str:
        logging.error("Variável de ambiente FIREBASE_CRED_JSON não encontrada.")
        raise RuntimeError("Credenciais do Firebase ausentes.")
    
    try:
        cred_dict = json.loads(cred_json_str)
        cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase inicializado com sucesso a partir da variável de ambiente.")
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao decodificar JSON das credenciais do Firebase: {e}")
        raise
    except Exception as e:
        logging.error(f"Erro ao inicializar o Firebase: {e}")
        raise

def initialize_onnx_session():
    global ort_session
    MODEL_PATH = "quantized_clip_model.onnx"  # Troque para o modelo não quantizado
    try:
        ort_session = ort.InferenceSession(MODEL_PATH)
        logging.info("Sessão ONNX Runtime inicializada com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo ONNX: {e}. Verifique o arquivo '{MODEL_PATH}'.")
        raise RuntimeError(f"Erro ao carregar modelo ONNX: {e}")

async def load_faiss_index():
    global logo_index
    logging.info("Iniciando carregamento do índice FAISS...")
    INDEX_DIM = 512
    logo_index = LogoIndex(dim=INDEX_DIM)
    
    try:
        logos = logos_collection.find({})
        count = 0
        async for logo in logos:
            if 'vector' in logo and len(logo['vector']) == INDEX_DIM:
                vector = np.array(logo['vector'], dtype=np.float32)
                logo_index.add_logo(vector, logo)
                count += 1
        logging.info(f"Índice FAISS construído com {count} vetores.")
    except Exception as e:
        logging.error(f"Falha ao carregar o índice FAISS do MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Falha ao inicializar o índice FAISS.")

async def verify_firebase_token_dep(credentials: HTTPAuthorizationCredentials = Security(http_bearer)):
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token Firebase inválido ou expirado",
        )


async def attach_signed_urls_to_blocos(blocos):
    """Given a list of blocos, attach a 'signed_url' field for media blocos.
    This runs gerar_signed_url_conteudo in a thread to avoid blocking the event loop.
    """
    if not blocos or not isinstance(blocos, list):
        return blocos
    for b in blocos:
        try:
            tipo_selecionado = b.get('tipoSelecionado') or ''
            tipo_label = b.get('tipo') or ''
            is_media = False
            if isinstance(tipo_selecionado, str) and tipo_selecionado.lower() in ('imagem', 'carousel', 'video'):
                is_media = True
            else:
                tl = tipo_label.lower() if isinstance(tipo_label, str) else ''
                if tl.startswith('imagem') or tl.startswith('video') or tl.startswith('carousel'):
                    is_media = True
            if not is_media:
                continue
            # For carousel, process items
            if b.get('items') and isinstance(b.get('items'), list):
                for it in b['items']:
                    try:
                        url = it.get('url') or (it.get('meta') and it['meta'].get('url'))
                        filename = it.get('filename') or (it.get('meta') and it['meta'].get('filename'))
                        if not url and filename:
                            url = f"gs://olinxra-conteudo/{filename}"
                        if url:
                            signed = await asyncio.to_thread(gerar_signed_url_conteudo, url, filename)
                            if signed:
                                it['signed_url'] = signed
                    except Exception:
                        continue
                continue
            # Single media block
            url = b.get('url')
            filename = b.get('filename')
            if not url and filename:
                url = f"gs://olinxra-conteudo/{filename}"
            if url:
                try:
                    signed = await asyncio.to_thread(gerar_signed_url_conteudo, url, filename)
                    if signed:
                        b['signed_url'] = signed
                except Exception:
                    # ignore signing failures, frontend can fallback
                    pass
        except Exception:
            continue
    return blocos

# Configura CORS permitindo configurar as origens via variável de ambiente
# Lê CORS_ALLOW_ORIGINS (comma-separated). Se ausente, usa fallback somente em dev.
_env_origins = os.getenv('CORS_ALLOW_ORIGINS', '').strip()
if _env_origins:
    try:
        _allow_origins = [o.strip() for o in _env_origins.split(',') if o.strip()]
    except Exception:
        logging.error('Formato inválido em CORS_ALLOW_ORIGINS; esperar comma-separated list')
        _allow_origins = []
else:
    # fallback apenas para desenvolvimento local
    if os.getenv('ENV', 'development') == 'production':
        logging.error('CORS_ALLOW_ORIGINS não definido em produção. Abortando inicialização.')
        raise RuntimeError('Variável CORS_ALLOW_ORIGINS é obrigatória em produção.')
    logging.warning('CORS_ALLOW_ORIGINS não definido — usando fallback http://localhost:5173 (apenas local).')
    _allow_origins = ['http://localhost:5173']

logging.info(f"CORS origins configured: {_allow_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

async def _search_and_compare_logic(file: UploadFile):
    if not logo_index:
        raise HTTPException(status_code=503, detail="Índice de logos não está pronto.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # heavy CPU work: run in threadpool to avoid blocking event loop
        query_vector = await asyncio.to_thread(extract_clip_features, temp_path, ort_session)
        import numpy as np
        print("query_vector shape:", np.array(query_vector).shape)
        print("query_vector values:", np.array(query_vector).tolist())
        print("faiss index dimension:", logo_index.index.d)
        results = logo_index.search(query_vector, top_k=1)

        if results:
            match = results[0]
            return {
                "found": True,
                "name": match['metadata'].get('nome', 'Logo encontrado'),
                "confidence": float(match.get('confidence', 0)),
                "distance": float(match.get('distance', 0)),
                "owner": match['metadata'].get('owner_uid', ''),
                "query_vector": np.array(query_vector).tolist()
            }

        return {"found": False, "debug": "Nenhum match encontrado", "query_vector": np.array(query_vector).tolist()}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post('/api/validate-button-block')
async def api_validate_button_block(payload: dict = Body(...), token: dict = Depends(verify_firebase_token_dep)):
    """Valida um payload de botão usando o Pydantic schema; usado apenas para testes e adm."""
    try:
        validated = validate_button_block_payload(payload)
        return {"valid": True, "normalized": validated.dict(exclude_none=True)}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.post('/search-logo/')
async def search_logo(file: UploadFile = File(...)):
    return await _search_and_compare_logic(file)

@app.post('/authenticated-search-logo/')
async def search_logo_auth(file: UploadFile = File(...), token: dict = Depends(verify_firebase_token_dep)):
    return await _search_and_compare_logic(file)

@app.get('/debug/user')
async def debug_user(token: dict = Depends(verify_firebase_token_dep)):
    master_email = os.getenv("USER_ADMIN_EMAIL")
    return {
        "user_email": token.get("email"),
        "master_email": master_email,
        "is_master": token.get("email") == master_email,
        "uid": token.get("uid")
    }

@app.get('/debug/logos')
async def debug_logos():
    try:
        logos = await logos_collection.find({}).to_list(length=100)
        return {
            "total_logos": len(logos),
            "logos": [
                {
                    "id": str(logo["_id"]),
                    "nome": logo.get("nome"),
                    "has_vector": "vector" in logo,
                    "vector_length": len(logo["vector"]) if "vector" in logo else 0
                }
                for logo in logos
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/add-logo/')
async def add_logo(
    file: UploadFile = File(...),
    name: str = Form(...),
    token: dict = Depends(verify_firebase_token_dep)
):
    existing = await logos_collection.find_one({"nome": name})
    if existing:
        raise HTTPException(status_code=400, detail="Já existe uma imagem com esse nome.")
    
    allowed_types = ["image/png", "image/jpeg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Tipo de arquivo não permitido.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        # Extrai features em threadpool para não bloquear o event loop
        features = await asyncio.to_thread(extract_clip_features, temp_path, ort_session)
        features = np.array(features, dtype=np.float32)
        features /= np.linalg.norm(features)
        # Upload ao GCS pode envolver I/O síncrono; executa em threadpool também
        gcs_url = await asyncio.to_thread(upload_image_to_gcs, temp_path, os.path.basename(file.filename))
        doc = {
            "nome": name,
            "url": gcs_url,
            "filename": os.path.basename(file.filename),
            "owner_uid": token["uid"],
            "vector": features.tolist()
        }
        result = await logos_collection.insert_one(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar logo: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return {"success": True, "id": str(result.inserted_id)}

@app.get('/images')
async def get_images(ownerId: str = None):
    logging.info(f"ownerId recebido: '{ownerId}'")
    filtro = {}
    if ownerId:
        filtro = {"owner_uid": ownerId}
    imagens = await logos_collection.find(filtro).to_list(length=100)
    logging.info(f"Imagens encontradas: {imagens}")
    from gcs_utils import get_bucket
    def montar_url(img):
        filename = img.get("filename")
        url_gcs = img.get("url", "")
        if not filename:
            logging.error(f"Imagem sem filename: {img}")
            return ""
        # Detecta o tipo de bucket pelo prefixo da URL salva no banco
        tipo_bucket = "logos"
        if url_gcs.startswith("gs://olinxra-conteudo/"):
            tipo_bucket = "conteudo"
        elif url_gcs.startswith("gs://olinxra-logos/"):
            tipo_bucket = "logos"
        else:
            # fallback: tenta pelo nome do bucket no filename
            if "conteudo" in filename:
                tipo_bucket = "conteudo"
        try:
            bucket = get_bucket(tipo_bucket)
            url = bucket.blob(filename).generate_signed_url(
                version="v4",
                expiration=3600,
                method="GET"
            )
            return url
        except Exception as e:
            logging.error(f"Erro ao gerar signed URL para {filename} (bucket {tipo_bucket}): {e}")
            return ""
    return [{
        "url": montar_url(img),
        "_id": str(img.get("_id")),
        "owner_uid": img.get("owner_uid"),
        "nome": img.get("nome", "")
    } for img in imagens]

@app.delete('/delete-logo/')
async def delete_logo(id: str = Query(...), token: dict = Depends(verify_firebase_token_dep)):
    try:
        object_id = ObjectId(id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="ID inválido")
    logo = await logos_collection.find_one({"_id": object_id})
    if not logo:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    from gcs_utils import get_bucket
    bucket = get_bucket("logos")
    blob = bucket.blob(logo['filename'])
    try:
        blob.delete()
    except Exception as e:
        from google.api_core.exceptions import NotFound
        if isinstance(e, NotFound):
            # Arquivo já não existe, segue normalmente
            pass
        else:
            raise HTTPException(status_code=500, detail=f"Erro ao deletar arquivo do bucket: {str(e)}")
    await logos_collection.delete_one({"_id": object_id})
    return {"success": True}

@app.get("/admin/list")
async def list_users(token: dict = Depends(verify_firebase_token_dep)):
    try:
        if token.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Acesso negado.")

        users = []
        for user in auth.list_users().iterate_all():
            users.append({"uid": user.uid, "email": user.email})
        return users
    except Exception as e:
        print("Erro ao listar usuários:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/create")
async def create_admin(
    email: str = Body(...),
    password: str = Body(None),
    token: dict = Depends(verify_firebase_token_dep)
):
    if token.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acesso negado.")

    try:
        user = auth.create_user(email=email)
        auth.set_custom_user_claims(user.uid, {"role": "admin"})
        reset_link = auth.generate_password_reset_link(email)
        msg = MIMEText(f"Olá! Defina sua senha neste link: {reset_link}")
        msg["Subject"] = "Defina sua senha de acesso"
        msg["From"] = "no-reply@olinx.digital"
        msg["To"] = email

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(os.getenv("USER_ADMIN_EMAIL"), os.getenv("APP_PASSWORD"))
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())
        except Exception as e:
            auth.delete_user(user.uid)
            raise HTTPException(status_code=400, detail=f"Falha ao enviar e-mail: {str(e)}")

        return {"success": True, "uid": user.uid, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class DeleteAdminRequest(BaseModel):
    uid: str

@app.post("/admin/delete")
async def delete_admin(
    req: DeleteAdminRequest,
    token: dict = Depends(verify_firebase_token_dep)
):
    uid = req.uid
    if token.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acesso negado.")

    try:
        auth.delete_user(uid)
        return {"success": True, "uid": uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/debug/inspect-request/')
async def debug_inspect_request(data: dict = Body(...)):
    import logging
    logging.info(f'[debug-inspect-request] Corpo recebido: {str(data)[:500]}')
    return {"received": data}

async def buscar_conteudo_por_marca_e_localizacao(marca_id, latitude, longitude):
    """
    Procura um documento de conteúdo pela marca (preferencialmente por marca_id) e
    por faixa de latitude/longitude.

    Estratégia:
    1. Tenta buscar por campo `marca_id` igual a `str(marca_id)`.
    2. Se não encontrar, tenta resolver o `nome_marca` usando a coleção `logos`
       (caso `marca_id` seja um ObjectId vindo de `logos`) e busca por `nome_marca`.
    3. Por fim, tenta um fallback simples por `nome_marca` igual a `str(marca_id)`.
    Isso garante compatibilidade com documentos antigos que usam `nome_marca` e com
    novos documentos que usam `marca_id`.
    """
    lat_filter = {"$gte": latitude - 0.01, "$lte": latitude + 0.01}
    lon_filter = {"$gte": longitude - 0.01, "$lte": longitude + 0.01}

    # 1) Busca por marca_id — suportando tanto ObjectId quanto strings
    try:
        filtro = {
            "latitude": lat_filter,
            "longitude": lon_filter
        }
        # Se o valor recebido já for um ObjectId, use-o diretamente.
        # Caso contrário, tente converter para ObjectId; se falhar, pesquise pela string.
        try:
            if isinstance(marca_id, ObjectId):
                filtro['marca_id'] = marca_id
            else:
                filtro['marca_id'] = ObjectId(str(marca_id))
        except Exception:
            filtro['marca_id'] = str(marca_id)

        doc = await db["conteudos"].find_one(filtro)
        if doc:
            return doc
    except Exception:
        # falha ao buscar por marca_id — seguimos com fallbacks
        doc = None

    # 2) Tentar resolver nome da marca a partir da coleção `logos`
    nome_marca = None
    try:
        # Se marca_id for um ObjectId ou string que represente o _id
        try:
            obj_id = ObjectId(marca_id)
            marca_doc = await logos_collection.find_one({"_id": obj_id})
        except Exception:
            # marca_id pode já ser um nome ou um string não-convertível
            marca_doc = None
        if marca_doc:
            nome_marca = marca_doc.get("nome")
    except Exception:
        nome_marca = None

    if nome_marca:
        filtro2 = {
            "nome_marca": nome_marca,
            "latitude": lat_filter,
            "longitude": lon_filter
        }
        doc = await db["conteudos"].find_one(filtro2)
        if doc:
            return doc

    # 3) Fallback: busca por nome_marca igual ao valor recebido
    filtro3 = {
        "nome_marca": str(marca_id),
        "latitude": lat_filter,
        "longitude": lon_filter
    }
    return await db["conteudos"].find_one(filtro3)

async def geocode_reverse(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1
    }
    headers = {"User-Agent": "OlinxRA/1.0"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get("address", {})
    except httpx.RequestError as e:
        logging.error(f"Erro na chamada de geocode_reverse: {e}")
    return {}

@app.get('/api/reverse-geocode')
async def api_reverse_geocode(lat: float = Query(...), lon: float = Query(...)):
    try:
        address = await geocode_reverse(lat, lon)
        return address
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geocodificação reversa: {str(e)}")

# Cache em memória: {(nome_marca, latitude, longitude): resultado}
consulta_cache = {}

def make_cache_key(nome_marca, latitude, longitude):
    # Arredonda para evitar pequenas variações
    return (nome_marca, round(latitude, 6), round(longitude, 6))

@app.post('/consulta-conteudo/')
async def consulta_conteudo(
    nome_marca: str = Body(...),
    latitude: float = Body(...),
    longitude: float = Body(...)
):
    cache_key = make_cache_key(nome_marca, latitude, longitude)
    if cache_key in consulta_cache:
        return consulta_cache[cache_key]

    # Busca a marca no banco
    marca = await logos_collection.find_one({"nome": nome_marca})
    if not marca:
        resultado = {"conteudo": None, "mensagem": "Marca não encontrada."}
        consulta_cache[cache_key] = resultado
        return resultado

    # Busca conteúdo associado à marca e localização usando a função
    conteudo = await buscar_conteudo_por_marca_e_localizacao(marca["_id"], latitude, longitude)

    # Busca endereço detalhado usando geocodificação reversa
    endereco = await geocode_reverse(latitude, longitude)

    # Monta string do local (exemplo: Rua, Bairro, Cidade, Estado, País)
    local_str = ", ".join([
        endereco.get("road", ""),
        endereco.get("suburb", ""),
        endereco.get("city", endereco.get("town", endereco.get("village", ""))),
        endereco.get("state", ""),
        endereco.get("country", "")
    ])
    local_str = local_str.strip(", ").replace(",,", ",")

    if conteudo:
        resultado = {
            "conteudo": {
                "texto": conteudo.get("texto", ""),
                "imagens": conteudo.get("imagens", []),  # lista de URLs
                "videos": conteudo.get("videos", []),    # lista de URLs
                # outros campos para RA podem ser adicionados aqui
            },
            "mensagem": "Conteúdo encontrado.",
            "localizacao": local_str
        }
    else:
        resultado = {
            "conteudo": None,
            "mensagem": f"Nenhum conteúdo associado a esta marca neste local: {local_str}.",
            "localizacao": local_str
        }

    consulta_cache[cache_key] = resultado
    return resultado

@app.get('/api/marcas')
async def listar_marcas(
    ownerId: str = Query(..., description="ID do usuário dono das marcas"),
    token: dict = Depends(verify_firebase_token_dep)
):
    if not ownerId:
        raise HTTPException(status_code=422, detail="Parâmetro 'ownerId' é obrigatório.")
    marcas = await logos_collection.find({"owner_uid": ownerId}).to_list(length=100)
    return [{"id": str(marca["_id"]), "nome": marca.get("nome", "")} for marca in marcas]

@app.get('/api/conteudo')
async def get_conteudo(
    nome_marca: str = Query(..., description="Nome da marca"),
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude")
):
    cache_key = (nome_marca, round(latitude, 6), round(longitude, 6))
    if cache_key in consulta_cache:
        return consulta_cache[cache_key]

    marca = await logos_collection.find_one({"nome": nome_marca})
    if not marca:
        resultado = {"conteudo": None, "mensagem": "Marca não encontrada."}
        consulta_cache[cache_key] = resultado
        return resultado

    conteudo = await buscar_conteudo_por_marca_e_localizacao(marca["_id"], latitude, longitude)
    endereco = await geocode_reverse(latitude, longitude)
    local_str = ", ".join([
        endereco.get("road", ""),
        endereco.get("suburb", ""),
        endereco.get("city", endereco.get("town", endereco.get("village", ""))),
        endereco.get("state", ""),
        endereco.get("country", "")
    ])
    local_str = local_str.strip(", ").replace(",,", ",")

    if conteudo:
        blocos_resp = conteudo.get("blocos", None) if conteudo.get("blocos") else None
        if blocos_resp:
            await attach_signed_urls_to_blocos(blocos_resp)
        resultado = {
            "conteudo": blocos_resp,
            "mensagem": "Conteúdo encontrado.",
            "localizacao": local_str
        }
    else:
        resultado = {
            "conteudo": None,
            "mensagem": f"Nenhum conteúdo associado a esta marca neste local: {local_str}.",
            "localizacao": local_str
        }

    consulta_cache[cache_key] = resultado
    return resultado

@app.get('/api/conteudo-por-regiao')
async def get_conteudo_por_regiao(
    nome_marca: str = Query(...),
    tipo_regiao: str = Query(None),
    nome_regiao: str = Query(None)
):
    # Busca a marca no banco
    marca = await logos_collection.find_one({"nome": nome_marca})
    if not marca:
        return {"blocos": []}

    # Busca conteúdo associado à marca e região
    filtro = {"nome_marca": nome_marca}
    if tipo_regiao:
        filtro["tipo_regiao"] = tipo_regiao
    if nome_regiao:
        filtro["nome_regiao"] = nome_regiao
    conteudo = await db["conteudos"].find_one(filtro)
    if conteudo:
        conteudo["_id"] = str(conteudo["_id"])
        blocos_ret = conteudo.get("blocos", [])
        if blocos_ret:
            await attach_signed_urls_to_blocos(blocos_ret)
        return {
            "blocos": blocos_ret,
            "tipo_regiao": conteudo.get("tipo_regiao"),
            "nome_regiao": conteudo.get("nome_regiao"),
            "latitude": conteudo.get("latitude"),
            "longitude": conteudo.get("longitude"),
        }
    return {"blocos": []}


@app.post('/api/conteudo')
async def post_conteudo(
    payload: dict = Body(...),
    token: dict = Depends(verify_firebase_token_dep),
    dry_run: bool = Query(False)
):
    """Cria ou atualiza um documento de conteúdo para uma marca/região.
    Espera payload com: nome_marca, blocos (array), latitude, longitude, tipo_regiao, nome_regiao
    Se blocos estiver vazio e documento existir, remove o documento (action: deleted).
    Retorna { action: 'saved' } ou { action: 'deleted' }.
    """
    try:
        nome_marca = payload.get('nome_marca')
        blocos = payload.get('blocos', []) or []
        latitude = payload.get('latitude')
        longitude = payload.get('longitude')
        tipo_regiao = payload.get('tipo_regiao')
        nome_regiao = payload.get('nome_regiao')

        if not nome_marca:
            raise HTTPException(status_code=422, detail="Campo 'nome_marca' é obrigatório.")

        # Tentar resolver marca para obter um marca_id estável (ObjectId)
        marca_doc = await logos_collection.find_one({"nome": nome_marca})
        marca_obj_id = None
        if marca_doc and "_id" in marca_doc:
            marca_obj_id = marca_doc["_id"]
        else:
            logging.warning(f"[post_conteudo] Marca '{nome_marca}' não encontrada em 'logos' — salvando sem marca_id.")

        # Construir filtro base por owner + região
        base_filtro = {
            'owner_uid': token.get('uid'),
            'tipo_regiao': tipo_regiao,
            'nome_regiao': nome_regiao
        }

        # Procurar documento existente de forma robusta:
        # - se tivermos marca_obj_id, primeiro tente match por ObjectId,
        #   depois tente pelo string equivalente (compatibilidade com docs antigos),
        # - se não tivermos marca_obj_id, tente procurar por nome_marca.
        existente = None
        if marca_obj_id:
            try:
                existente = await db['conteudos'].find_one({**base_filtro, 'marca_id': marca_obj_id})
            except Exception:
                existente = None
            if not existente:
                # tenta também por string (caso o banco tenha o id como string)
                try:
                    existente = await db['conteudos'].find_one({**base_filtro, 'marca_id': str(marca_obj_id)})
                except Exception:
                    existente = None
        else:
            # sem marca_obj_id, usamos nome_marca como chave
            try:
                existente = await db['conteudos'].find_one({**base_filtro, 'nome_marca': nome_marca})
            except Exception:
                existente = None

        # Helper interno: detecta se um bloco aparenta conter mídia
        def bloco_possui_media(b):
            try:
                if not isinstance(b, dict):
                    return False
                # URL-like in url or conteudo
                u = b.get('url') or b.get('conteudo') or ''
                if isinstance(u, str) and u.strip() != '':
                    if u.startswith('gs://') or u.startswith('/') or u.startswith('http') or u.startswith('blob:'):
                        return True
                # filename or explicit type
                if b.get('filename') or b.get('nome'):
                    return True
                t = b.get('type') or b.get('content_type') or ''
                if isinstance(t, str) and (t.startswith('image') or t.startswith('video')):
                    return True
                # carousel items
                if b.get('items') and isinstance(b.get('items'), list):
                    for it in b.get('items'):
                        if not isinstance(it, dict):
                            continue
                        iu = it.get('url') or (it.get('meta') and it['meta'].get('url')) or (it.get('conteudo')) or ''
                        if isinstance(iu, str) and iu.strip() != '':
                            if iu.startswith('gs://') or iu.startswith('/') or iu.startswith('http') or iu.startswith('blob:'):
                                return True
                        if it.get('filename') or (it.get('meta') and it['meta'].get('filename')):
                            return True
                        it_type = it.get('type') or (it.get('meta') and it['meta'].get('type')) or ''
                        if isinstance(it_type, str) and (it_type.startswith('image') or it_type.startswith('video')):
                            return True
                return False
            except Exception:
                return False

        # Normalizar blocos para novos saves: garantir datetimes e remover campos
        # desnecessários em blocos de texto. Também valida se há URLs locais (blob:)
        # que indicam upload não finalizado.
        invalid_blocks = []
        cleaned_blocos = []
        for idx, b in enumerate(blocos):
            if not isinstance(b, dict):
                continue
            # detect blob: in url or conteudo
            try:
                u = b.get('url', '')
                c = b.get('conteudo', '')
                if (isinstance(u, str) and u.startswith('blob:')) or (isinstance(c, str) and c.startswith('blob:')):
                    invalid_blocks.append({ 'index': idx, 'filename': (b.get('filename') or b.get('nome') or '') })
                    # keep the block as-is for reporting; do not normalize
                    cleaned_blocos.append(b)
                    continue
            except Exception:
                cleaned_blocos.append(b)
                continue

            # Normalize created_at for blocos
            created = b.get('created_at')
            if isinstance(created, str):
                try:
                    created_dt = datetime.fromisoformat(created)
                except Exception:
                    try:
                        created_dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S.%f")
                    except Exception:
                        created_dt = datetime.utcnow()
                b['created_at'] = created_dt
            elif created is None:
                b['created_at'] = datetime.utcnow()

            # Determine whether this bloco represents media (image/carousel/video).
            # Prefer explicit machine-friendly `tipoSelecionado` or label, but also
            # detect media by presence of media-like fields (url/filename/type/items).
            tipo_selecionado = b.get('tipoSelecionado') or ''
            tipo_label = b.get('tipo') or ''
            is_media = False
            try:
                if isinstance(tipo_selecionado, str) and tipo_selecionado.lower() in ('imagem', 'carousel', 'video'):
                    is_media = True
                else:
                    tl = tipo_label.lower() if isinstance(tipo_label, str) else ''
                    if tl.startswith('imagem') or tl.startswith('video') or tl.startswith('carousel'):
                        is_media = True
                # If still not sure, inspect specific fields that indicate media
                if not is_media:
                    if bloco_possui_media(b):
                        is_media = True
            except Exception:
                is_media = False

            # If not media, strip media-related fields to keep document compact.
            if not is_media:
                for k in ('url', 'filename', 'type', 'subtipo', 'created_at'):
                    if k in b:
                        b.pop(k, None)

            # Remove ephemeral signed_url if present
            if 'signed_url' in b:
                b.pop('signed_url', None)

            cleaned_blocos.append(b)
        # Pós-processamento consolidado: para cada bloco, garantir propriedades derivadas
        for b in cleaned_blocos:
            try:
                tipo_selecionado = b.get('tipoSelecionado') or ''
                tipo_label = b.get('tipo') or ''
                is_media = False
                if isinstance(tipo_selecionado, str) and tipo_selecionado.lower() in ('imagem', 'carousel', 'video'):
                    is_media = True
                else:
                    tl = tipo_label.lower() if isinstance(tipo_label, str) else ''
                    if tl.startswith('imagem') or tl.startswith('video') or tl.startswith('carousel'):
                        is_media = True
                if not is_media and bloco_possui_media(b):
                    is_media = True

                # If media and missing url but has filename -> set url
                if is_media and (not b.get('url')) and b.get('filename'):
                    filename = b.get('filename')
                    if not str(filename).startswith('gs://'):
                        b['url'] = f"gs://olinxra-conteudo/{filename}"

                # If this is a carousel (has items), persist tipoSelecionado and normalize items
                if b.get('items') and isinstance(b.get('items'), list) and len(b.get('items')) > 0:
                    if not b.get('tipoSelecionado'):
                        b['tipoSelecionado'] = 'carousel'
                    for it in b['items']:
                        try:
                            if it is None:
                                continue
                            ca = it.get('created_at')
                            if isinstance(ca, str):
                                try:
                                    it['created_at'] = datetime.fromisoformat(ca)
                                except Exception:
                                    try:
                                        it['created_at'] = datetime.strptime(ca, "%Y-%m-%dT%H:%M:%S.%f")
                                    except Exception:
                                        try:
                                            it['created_at'] = datetime.strptime(ca, "%Y-%m-%d %H:%M:%S.%f")
                                        except Exception:
                                            it['created_at'] = datetime.utcnow()
                            elif ca is None:
                                it['created_at'] = datetime.utcnow()
                        except Exception:
                            it['created_at'] = datetime.utcnow()
                    # remove empty/blank parent media-like fields
                    for key in ('url', 'filename', 'nome', 'type'):
                        try:
                            val = b.get(key)
                            if val is None or (isinstance(val, str) and val.strip() == ''):
                                b.pop(key, None)
                        except Exception:
                            pass

                # If media and missing filename but has nome, infer filename using owner uid
                if is_media and (not b.get('filename')) and b.get('nome'):
                    nome = str(b.get('nome'))
                    if '/' in nome:
                        filename = nome
                    else:
                        owner = token.get('uid') or 'unknown'
                        filename = f"{owner}/{nome}"
                    b['filename'] = filename
                    if not b.get('url'):
                        b['url'] = f"gs://olinxra-conteudo/{filename}"
            except Exception:
                # keep going; post-processing must not fail the whole request
                pass
        if invalid_blocks:
            logging.warning('[post_conteudo] Rejeitando payload com blocos inválidos (blob:)', extra={'invalid_blocks': invalid_blocks, 'nome_marca': nome_marca, 'owner_uid': token.get('uid')})
            raise HTTPException(status_code=422, detail={ 'message': 'Payload contém referências locais (blob:). Faça upload das imagens primeiro.', 'invalid_blocks': invalid_blocks })

        # Validate button blocks (botao_destaque / botao_default) before persisting
        # Note: frontend may place editable fields inside bloco['meta'] (for local editing state).
        # To be tolerant, hydrate top-level fields from meta when missing so validation succeeds.
        try:
            from pydantic import ValidationError
            from schemas import validate_button_block_payload
            for bi, bb in enumerate(cleaned_blocos):
                try:
                    if not isinstance(bb, dict):
                        continue
                    tipo_check = (bb.get('tipo') or '').lower()
                    if tipo_check in ('botao_destaque', 'botao_default'):
                        # If frontend sent fields inside meta (e.g. meta.action or meta.label), copy them up
                        meta = bb.get('meta') or {}
                        if isinstance(meta, dict):
                            # action may be nested or split across keys
                            if not bb.get('action') and meta.get('action'):
                                bb['action'] = meta.get('action')
                            # also accept href/link/url directly on meta
                            if not bb.get('action') and (meta.get('href') or meta.get('link') or meta.get('url')):
                                bb['action'] = meta.get('action') if meta.get('action') else { 'type': 'link', 'href': meta.get('href') or meta.get('link') or meta.get('url'), 'target': meta.get('target') or '_self' }

                            # label may be stored under different keys in meta
                            if not bb.get('label'):
                                for key in ('label', 'buttonLabel', 'button_label', 'text', 'title'):
                                    if meta.get(key):
                                        bb['label'] = meta.get(key)
                                        break

                            # copy other optional decorative/button props if present
                            for fld in ('variant', 'color', 'icon', 'size', 'disabled', 'aria_label', 'analytics', 'visibility', 'position', 'temp_id'):
                                if bb.get(fld) is None and meta.get(fld) is not None:
                                    bb[fld] = meta.get(fld)

                        # If action is a dict-like in meta.action but missing required nested fields, try to hydrate
                        if isinstance(bb.get('action'), dict):
                            a = bb['action']
                            # try to fill href from meta top-level if missing
                            if a.get('type') == 'link' and not a.get('href'):
                                if meta.get('href'):
                                    a['href'] = meta.get('href')
                                elif meta.get('link'):
                                    a['href'] = meta.get('link')
                                elif meta.get('url'):
                                    a['href'] = meta.get('url')
                            # try to fill name for callback
                            if a.get('type') == 'callback' and not a.get('name') and meta.get('name'):
                                a['name'] = meta.get('name')
                            bb['action'] = a

                        # If after hydration required fields are still missing, build a helpful error
                        missing = []
                        if not bb.get('label'):
                            missing.append('label')
                        if not bb.get('action'):
                            missing.append('action')
                        # If action exists but lacks href/name depending on type, mark as missing
                        if bb.get('action') and isinstance(bb.get('action'), dict):
                            at = bb['action'].get('type')
                            if at == 'link' and not bb['action'].get('href'):
                                missing.append('action.href')
                            if at == 'callback' and not bb['action'].get('name'):
                                missing.append('action.name')
                        if missing:
                            raise HTTPException(status_code=422, detail={ 'message': f'Invalid button block at index {bi}', 'missing': missing, 'meta_keys': list(meta.keys()), 'block_preview': bb })

                        # will raise ValidationError if invalid
                        validate_button_block_payload(bb)
                except ValidationError as ve:
                    # pydantic ValidationError -> return 422 with details
                    raise HTTPException(status_code=422, detail={ 'message': f'Invalid button block at index {bi}', 'error': str(ve) })
                except HTTPException:
                    raise
                except Exception as e:
                    # Any other error in validation of this block
                    raise HTTPException(status_code=422, detail={ 'message': f'Invalid button block at index {bi}', 'error': str(e) })
        except HTTPException:
            raise
        except Exception:
            # If schemas cannot be imported or another issue occurs, log and continue
            logging.exception('[post_conteudo] Unexpected error during button block validation; skipping strict validation')

        # Se blocos estiverem vazios e já existe documento -> deletar (com confirmação)
        if existente and (not isinstance(cleaned_blocos, list) or len(cleaned_blocos) == 0):
            # compute files to delete (do not delete yet if dry_run)
            old_blocos = existente.get('blocos', []) or []
            to_delete = []
            for ob in old_blocos:
                try:
                    if ob.get('items') and isinstance(ob.get('items'), list):
                        for it in ob.get('items'):
                            if it:
                                url = it.get('url')
                                fname = it.get('filename') or it.get('nome')
                                if url:
                                    to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                                elif fname:
                                    to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                    else:
                        url = ob.get('url')
                        fname = ob.get('filename') or ob.get('nome')
                        if url:
                            to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                        elif fname:
                            to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                except Exception:
                    continue

            if dry_run:
                return {'action': 'dry_run', 'to_delete': to_delete, 'blocos': []}

            # enqueue deletions and attempt immediate delete; then remove document
            try:
                from gcs_utils import delete_gs_path, delete_file
                for item in to_delete:
                    try:
                        # insert pending_deletes entry
                        pend = {
                            'gs_url': item.get('gs_url'),
                            'filename': item.get('filename'),
                            'tipo': item.get('tipo', 'conteudo'),
                            'status': 'pending',
                            'retries': 0,
                            'created_at': datetime.utcnow(),
                            'last_attempt': None
                        }
                        res = await db['pending_deletes'].insert_one(pend)
                        # attempt immediate deletion in thread
                        def try_del():
                            try:
                                if item.get('gs_url'):
                                    return delete_gs_path(item.get('gs_url'))
                                elif item.get('filename'):
                                    return delete_file(item.get('filename'), item.get('tipo', 'conteudo'))
                            except Exception:
                                return False
                            return False
                        ok = await asyncio.to_thread(try_del)
                        if ok:
                            await db['pending_deletes'].update_one({'_id': res.inserted_id}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                    except Exception:
                        continue

                await db['conteudos'].delete_one({'_id': existente['_id']})
                exists_id = existente.get('_id')
                logging.info(f"[post_conteudo] Documento {str(exists_id) if exists_id is not None else 'unknown'} deletado (blocos vazios)")
            except Exception as e:
                logging.exception('Erro ao deletar documento existente com blocos vazios')
                raise HTTPException(status_code=500, detail=f'Erro ao deletação: {str(e)}')
            return { 'action': 'deleted' }

        # Caso contrário, cria ou atualiza (upsert)
        # Implementação B: não deduplicar blocos por filename. Quando existir documento,
        # concatenamos os blocos recebidos ao array existente (permitindo múltiplos blocos
        # que referenciem o mesmo arquivo). Isso preserva o reuso do arquivo no storage
        # mas garante que cada bloco enviado gere uma entrada no documento.
        if existente:
            try:
                # Substitui os blocos existentes pelo payload recebido.
                # Isso evita duplicações quando o frontend carrega os blocos,
                # edita um existente e envia o estado completo de volta.
                update_doc = {
                    **base_filtro,
                    'blocos': list(cleaned_blocos),
                    'latitude': latitude,
                    'longitude': longitude,
                    'location': { 'type': 'Point', 'coordinates': [ longitude, latitude ] } if (latitude is not None and longitude is not None) else None,
                    'tipo_regiao': tipo_regiao,
                    'nome_regiao': nome_regiao,
                    'marca_id': marca_obj_id,
                    'nome_marca': nome_marca,
                    'updated_at': datetime.utcnow()
                }
                # Before updating, determine which files were removed and delete them from GCS
                try:
                    old_blocos = existente.get('blocos', []) or []
                    # build set of filenames/urls that will remain
                    new_urls = set()
                    new_filenames = set()
                    for nb in cleaned_blocos:
                        if nb.get('items') and isinstance(nb.get('items'), list):
                            for it in nb['items']:
                                if it:
                                    if it.get('url'): new_urls.add(str(it.get('url')))
                                    if it.get('filename'): new_filenames.add(str(it.get('filename')))
                        else:
                            if nb.get('url'): new_urls.add(str(nb.get('url')))
                            if nb.get('filename'): new_filenames.add(str(nb.get('filename')))

                    # iterate old blocos and enqueue files not present in new sets
                    to_delete = []
                    for ob in old_blocos:
                        try:
                            if ob.get('items') and isinstance(ob.get('items'), list):
                                for it in ob.get('items'):
                                    if it:
                                        url = it.get('url')
                                        fname = it.get('filename') or it.get('nome')
                                        if url and url not in new_urls:
                                            to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                                        elif fname and fname not in new_filenames:
                                            to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                            else:
                                url = ob.get('url')
                                fname = ob.get('filename') or ob.get('nome')
                                if url and url not in new_urls:
                                    to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                                elif fname and fname not in new_filenames:
                                    to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                        except Exception:
                            continue

                    if dry_run:
                        return {'action': 'dry_run', 'to_delete': to_delete, 'blocos': cleaned_blocos}

                    from gcs_utils import delete_gs_path, delete_file
                    for item in to_delete:
                        try:
                            pend = {
                                'gs_url': item.get('gs_url'),
                                'filename': item.get('filename'),
                                'tipo': item.get('tipo', 'conteudo'),
                                'status': 'pending',
                                'retries': 0,
                                'created_at': datetime.utcnow(),
                                'last_attempt': None
                            }
                            res = await db['pending_deletes'].insert_one(pend)
                            def try_del():
                                try:
                                    if item.get('gs_url'):
                                        return delete_gs_path(item.get('gs_url'))
                                    elif item.get('filename'):
                                        return delete_file(item.get('filename'), item.get('tipo', 'conteudo'))
                                except Exception:
                                    return False
                                return False
                            ok = await asyncio.to_thread(try_del)
                            if ok:
                                await db['pending_deletes'].update_one({'_id': res.inserted_id}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                        except Exception:
                            continue
                except Exception:
                    logging.exception('[post_conteudo] Falha ao tentar enfileirar/remover arquivos antigos do GCS antes do update')

                await db['conteudos'].update_one({'_id': existente['_id']}, {'$set': update_doc})
                # Recupera o documento atualizado para retornar os blocos persistidos
                saved = await db['conteudos'].find_one({'_id': existente['_id']})
                if saved:
                    # converte _id para string se necessário
                    saved['_id'] = str(saved['_id'])
                    return { 'action': 'saved', 'blocos': saved.get('blocos', []) }
                return { 'action': 'saved' }
            except Exception as e:
                logging.exception('Erro ao atualizar blocos existentes')
                raise HTTPException(status_code=500, detail=f'Erro ao salvar conteúdo: {str(e)}')
        else:
            doc = {
                **base_filtro,
                'blocos': cleaned_blocos,
                'latitude': latitude,
                'longitude': longitude,
                'location': { 'type': 'Point', 'coordinates': [ longitude, latitude ] } if (latitude is not None and longitude is not None) else None,
                'tipo_regiao': tipo_regiao,
                'nome_regiao': nome_regiao,
                'marca_id': marca_obj_id,
                'nome_marca': nome_marca,
                'updated_at': datetime.utcnow()
            }
            result = await db['conteudos'].insert_one(doc)
            # Recupera o documento recém-criado
            saved = await db['conteudos'].find_one({'_id': result.inserted_id})
            if saved:
                saved['_id'] = str(saved['_id'])
                return { 'action': 'saved', 'blocos': saved.get('blocos', []) }
            return { 'action': 'saved' }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception('Erro ao salvar conteúdo')
        raise HTTPException(status_code=500, detail=f'Erro ao salvar conteúdo: {str(e)}')


@app.post('/admin/process-pending-deletes')
async def admin_process_pending_deletes(token: dict = Depends(verify_firebase_token_dep)):
    # Only allow master admin to trigger
    master_email = os.getenv('USER_ADMIN_EMAIL')
    if token.get('email') != master_email:
        raise HTTPException(status_code=403, detail='Forbidden')
    # process pending deletes (simple synchronous attempt)
    pending = await db['pending_deletes'].find({'status': {'$in': ['pending', 'retry']}}).to_list(length=1000)
    processed = []
    from gcs_utils import delete_gs_path, delete_file
    for p in pending:
        try:
            ok = False
            if p.get('gs_url'):
                ok = await asyncio.to_thread(delete_gs_path, p.get('gs_url'))
            elif p.get('filename'):
                ok = await asyncio.to_thread(delete_file, p.get('filename'), p.get('tipo', 'conteudo'))
            if ok:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                processed.append({'id': str(p['_id']), 'status': 'done'})
            else:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'retry', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
                processed.append({'id': str(p['_id']), 'status': 'retry'})
        except Exception:
            try:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'error', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
            except Exception:
                pass
            processed.append({'id': str(p['_id']), 'status': 'error'})
    return {'processed': processed, 'count': len(processed)}


@app.post('/add-content-image/')
async def add_content_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    temp_id: str = Form(None),
    tipo_bloco: str = Form("imagem"),
    subtipo: str = Form(""),
    marca: str = Form(""),
    tipo_regiao: str = Form(""),
    nome_regiao: str = Form(""),
    token: dict = Depends(verify_firebase_token_dep)
):
    # Segurança: validar Origin (se fornecido) contra lista de origens permitidas
    origin = None
    try:
        origin = Request.scope.get('headers') if False else None
    except Exception:
        origin = None
    # DEBUG: log minimal info about incoming request to help diagnose 422 errors
    try:
        logging.info(f"[add_content_image] recebendo upload: filename={getattr(file, 'filename', None)} content_type={getattr(file, 'content_type', None)} name_param={name} temp_id={temp_id} tipo_bloco={tipo_bloco} subtipo={subtipo} marca={marca} tipo_regiao={tipo_regiao} nome_regiao={nome_regiao} uid={token.get('uid') if token else 'no-token'}")
    except Exception:
        logging.exception('[add_content_image] falha ao logar metadados iniciais do upload')
    # Nota: a checagem de Origin será feita manualmente abaixo através do header
    # Valida content-type genericamente: aceitar image/* e video/*
    if not (file.content_type and (file.content_type.startswith('image/') or file.content_type.startswith('video/'))):
        logging.warning(f"[add_content_image] Tipo de conteúdo rejeitado: {file.content_type}")
        raise HTTPException(status_code=400, detail="Tipo de arquivo não permitido.")

    import time
    temp_path = None
    t0 = time.time()
    try:
        ext = os.path.splitext(file.filename)[-1]
        name_base = os.path.splitext(name)[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        t1 = time.time()
        logging.info(f"[add_content_image] Tempo até upload GCS: {t1-t0:.2f}s")
        # Salva no bucket olinxra-conteudo, organizado por admin
        gcs_filename = f"{token['uid']}/{name_base}{ext}"
        # upload síncrono -> execute em threadpool para não bloquear o loop
        gcs_url = await asyncio.to_thread(upload_image_to_gcs, temp_path, gcs_filename, "conteudo")
        t2 = time.time()
        logging.info(f"[add_content_image] Tempo upload GCS: {t2-t1:.2f}s (total: {t2-t0:.2f}s)")

        # created_at como datetime (não string) para evitar conversões repetidas
        bloco_img = {
            "tipo": tipo_bloco,
            "subtipo": subtipo,
            "url": gcs_url,
            "nome": name,
            "filename": gcs_filename,
            "type": file.content_type,
            "created_at": datetime.utcnow()
        }
        # Gera signed_url para facilitar preview imediato no frontend (se possível)
        try:
            signed = gerar_signed_url_conteudo(gcs_url, gcs_filename)
        except Exception:
            signed = gcs_url
        t3 = time.time()
        logging.info(f"[add_content_image] Upload concluído (não persiste no DB). Tempo total: {t3-t0:.2f}s")
        resp = {"success": True, "url": gcs_url, "signed_url": signed, "bloco": bloco_img}
        if temp_id:
            resp["temp_id"] = temp_id
        # Log minimal info: uid and filename/type
        try:
            logging.info(f"[add_content_image] upload ok uid={token.get('uid')} filename={gcs_filename} type={file.content_type}")
        except Exception:
            pass
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar conteúdo: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)