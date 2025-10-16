import logging
import json
import os
import numpy as np
import firebase_admin
from fastapi import Request
from firebase_admin import credentials, auth
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form, Query, Body, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from bson import ObjectId
from bson.errors import InvalidId
from datetime import timedelta
from gcs_utils import upload_image_to_gcs
from clip_utils import extract_clip_features
from faiss_index import LogoIndex
import tempfile
import smtplib
from email.mime.text import MIMEText
import asyncio
import onnxruntime as ort
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()

logo_index = None
ort_session = None
http_bearer = HTTPBearer()
client = None
db = None
logos_collection = None
images_collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db, logos_collection, images_collection
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise RuntimeError("Variável de ambiente MONGO_URI não encontrada.")
    DB_NAME = os.getenv("MONGO_DB_NAME", "olinxra")

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    logos_collection = db["logos"]
    images_collection = db["images"]

    logging.info("Iniciando a aplicação...")
    initialize_firebase()
    initialize_onnx_session()
    await load_faiss_index()
    yield
    if client:
        client.close()

app = FastAPI(lifespan=lifespan)

# --- Helpers de Inicialização ---
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

        query_vector = extract_clip_features(temp_path, ort_session)
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
        features = extract_clip_features(temp_path, ort_session)
        features = np.array(features, dtype=np.float32)
        features /= np.linalg.norm(features)
        gcs_url = upload_image_to_gcs(temp_path, os.path.basename(file.filename))
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
    def montar_url(img):
        filename = img.get("filename")
        if not filename:
            return ""
        try:
            # Gera uma signed URL válida por 1 hora
            url = bucket.blob(filename).generate_signed_url(
                version="v4",
                expiration=3600,
                method="GET"
            )
            return url
        except Exception as e:
            logging.error(f"Erro ao gerar signed URL para {filename}: {e}")
            return ""
    return [{"url": montar_url(img), "_id": str(img.get("_id")), "owner_uid": img.get("owner_uid"), "nome": img.get("nome", "")} for img in imagens]

@app.delete('/delete-logo/')
async def delete_logo(id: str = Query(...), token: dict = Depends(verify_firebase_token_dep)):
    try:
        object_id = ObjectId(id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="ID inválido")
    logo = await logos_collection.find_one({"_id": object_id})
    if not logo:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
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
    return await db["conteudos"].find_one({
        "marca_id": str(marca_id),
        "latitude": {"$gte": latitude - 0.01, "$lte": latitude + 0.01},
        "longitude": {"$gte": longitude - 0.01, "$lte": longitude + 0.01}
    })

async def geocode_reverse(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1
    }
    headers = {"User-Agent": "OlinxRA/1.0"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json().get("address", {})
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
        resultado = {
            "conteudo": conteudo.get("blocos", None) if conteudo.get("blocos") else None,
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
        return {
            "blocos": conteudo.get("blocos", []),
            "tipo_regiao": conteudo.get("tipo_regiao"),
            "nome_regiao": conteudo.get("nome_regiao"),
            "latitude": conteudo.get("latitude"),
            "longitude": conteudo.get("longitude"),
        }
    return {"blocos": []}

@app.post('/api/conteudo')
async def create_conteudo(request: Request):
    data = await request.json()
    print("[POST /api/conteudo] Dados recebidos:", data)
    nome_marca = data.get("nome_marca")
    blocos = data.get("blocos", [])
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    tipo_regiao = data.get("tipo_regiao")
    nome_regiao = data.get("nome_regiao")
    if not nome_marca or latitude is None or longitude is None:
        raise HTTPException(status_code=422, detail="Parâmetros obrigatórios ausentes.")

    marca = await logos_collection.find_one({"nome": nome_marca})
    if not marca:
        raise HTTPException(status_code=404, detail="Marca não encontrada.")

    lat_rounded = round(float(latitude), 6)
    lon_rounded = round(float(longitude), 6)
    conteudo_doc = {
        "marca_id": str(marca["_id"]),
        "nome_marca": nome_marca,
        "latitude": lat_rounded,
        "longitude": lon_rounded,
        "tipo_regiao": tipo_regiao,
        "nome_regiao": nome_regiao,
        "blocos": blocos,
    }
    # Se blocos está vazio, exclui o documento correspondente
    if not blocos:
        delete_result = await db["conteudos"].delete_one({
            "marca_id": str(marca["_id"]),
            "tipo_regiao": tipo_regiao,
            "nome_regiao": nome_regiao
        })
        return {
            "success": True,
            "action": "deleted",
            "deleted": delete_result.deleted_count
        }
    # Upsert: atualiza se já existe, senão cria
    result = await db["conteudos"].update_one(
        {
            "marca_id": str(marca["_id"]),
            "tipo_regiao": tipo_regiao,
            "nome_regiao": nome_regiao
        },
        {"$set": conteudo_doc},
        upsert=True
    )
    return {
        "success": True,
        "action": "saved",
        "conteudo_id": str(result.upserted_id) if result.upserted_id else "updated"
    }

@app.post('/add-content-image/')
async def add_content_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    token: dict = Depends(verify_firebase_token_dep)
):
    allowed_types = ["image/png", "image/jpeg", "video/mp4"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Tipo de arquivo não permitido.")

    temp_path = None
    try:
        ext = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        # Salva no bucket olinxra-conteudo, organizado por admin
        gcs_filename = f"conteudo/{token['uid']}/{name}{ext}"
        gcs_url = upload_image_to_gcs(temp_path, gcs_filename)
        # Salva referência no banco
        doc = {
            "nome": name,
            "url": gcs_url,
            "filename": gcs_filename,
            "owner_uid": token["uid"],
            "type": file.content_type
        }
        result = await images_collection.insert_one(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar conteúdo: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return {"success": True, "id": str(result.inserted_id), "url": gcs_url}