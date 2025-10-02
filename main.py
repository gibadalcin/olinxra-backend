import logging
import json
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form, Query, Body, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from bson import ObjectId
from bson.errors import InvalidId
from datetime import timedelta
from gcs_utils import upload_image_to_gcs, bucket
from clip_utils import extract_clip_features
from faiss_index import LogoIndex
import tempfile
import smtplib
from email.mime.text import MIMEText
import asyncio
import onnxruntime as ort
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()


from contextlib import asynccontextmanager

logo_index = None
ort_session = None
http_bearer = HTTPBearer()
client = None
db = None
logos_collection = None

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
        print("faiss index dimension:", logo_index.index.d)
        results = logo_index.search(query_vector, top_k=1)
        
        if results:
            match = results[0]
            return {
                "found": True,
                "name": match['metadata'].get('nome', 'Logo encontrado'),
                "confidence": float(match['distance']),
                "owner": match['metadata'].get('owner_uid', '')
            }
        
        return {"found": False, "debug": "Nenhum match encontrado"}
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post('/search-logo/')
async def search_logo_public(file: UploadFile = File(...)):
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
async def get_images(token: dict = Depends(verify_firebase_token_dep)):
    master_email = os.getenv("USER_ADMIN_EMAIL")
    if token.get("email") == master_email:
        logos = await logos_collection.find({}).to_list(length=1000)
    else:
        logos = await logos_collection.find({"owner_uid": token["uid"]}).to_list(length=1000)
    
    imagens = []
    for logo in logos:
        blob = bucket.blob(logo['filename'])
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET"
        )
        imagens.append({
            "_id": str(logo["_id"]),
            "nome": logo.get("nome", ""),
            "url": signed_url,
            "owner_uid": logo.get("owner_uid", ""),
            "owner_email": logo.get("owner_email", "")
        })
    return imagens

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
    blob.delete()
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