import os
import json
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

# Nomes das variáveis de ambiente para o GCS
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET")
GCS_CREDENTIALS_JSON = os.getenv("CLOUD_STORAGE_CRED_JSON")

if not GCS_CREDENTIALS_JSON:
    # Se a variável de ambiente não estiver definida (como em testes locais)
    # o código irá falhar aqui. Em produção, ele deve estar presente.
    raise RuntimeError("Variável de ambiente CLOUD_STORAGE_CRED_JSON não encontrada.")

if GCS_CREDENTIALS_JSON.startswith('"') and GCS_CREDENTIALS_JSON.endswith('"'):
    GCS_CREDENTIALS_JSON = GCS_CREDENTIALS_JSON[1:-1]  # Remove aspas duplas externas
GCS_CREDENTIALS_JSON = GCS_CREDENTIALS_JSON.replace('\\"', '"')  # Corrige aspas escapadas

try:
    # Carrega o conteúdo JSON da variável de ambiente
    cred_dict = json.loads(GCS_CREDENTIALS_JSON)
    
    # CORREÇÃO: Desfaz o escape da chave privada para o formato PEM,
    # caso ela tenha sido colada em linha única na variável de ambiente.
    if 'private_key' in cred_dict:
        cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
    
    # Cria o cliente a partir do dicionário de credenciais
    storage_client = storage.Client.from_service_account_info(cred_dict)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Erro ao decodificar JSON das credenciais do GCS: {e}")

# Conecta ao bucket
try:
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
except Exception as e:
    raise RuntimeError(f"Erro ao conectar ao bucket '{GCS_BUCKET_NAME}': {e}")


def upload_image_to_gcs(local_path, filename):
    """
    Faz o upload de um arquivo local para o Google Cloud Storage.
    """
    blob = bucket.blob(filename)
    blob.upload_from_filename(local_path)
    # Retorna o URL do Google Cloud Storage
    gcs_url = f"gs://{bucket.name}/{filename}"
    return gcs_url