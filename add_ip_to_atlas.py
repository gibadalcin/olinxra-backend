import requests
import os
import sys
from requests.auth import HTTPDigestAuth # Importa a autenticação Digest
from dotenv import load_dotenv

# Carrega variáveis do .env (Apenas para desenvolvimento local, DigitalOcean usa ENV vars diretas)
load_dotenv()

# Configurações do Atlas via ENV
ATLAS_PUBLIC_KEY = os.getenv('ATLAS_PUBLIC_KEY')
ATLAS_PRIVATE_KEY = os.getenv('ATLAS_PRIVATE_KEY')
ATLAS_PROJECT_ID = os.getenv('ATLAS_PROJECT_ID')

if not ATLAS_PUBLIC_KEY or not ATLAS_PRIVATE_KEY or not ATLAS_PROJECT_ID:
    print("[ERRO] Variáveis ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY ou ATLAS_PROJECT_ID não encontradas!")
    sys.exit(1)

# 1. Descobre o IP público atual
try:
    ip_response = requests.get('https://api.ipify.org', timeout=5)
    ip_response.raise_for_status()
    ip = ip_response.text.strip()
    print(f'IP público detectado: {ip}')
except requests.exceptions.RequestException as e:
    print(f'[ERRO] Falha ao obter o IP público: {e}')
    sys.exit(1)

# 2. Adiciona o IP à whitelist do Atlas
# CORREÇÃO DA URI: A API espera uma LISTA de objetos JSON.
url = f'https://cloud.mongodb.com/api/atlas/v1.0/groups/{ATLAS_PROJECT_ID}/accessList'
data = [
    {
        "ipAddress": ip,
        "comment": "IP do backend DigitalOcean (adicionado automaticamente)"
    }
]

# CORREÇÃO CRÍTICA: Usa HTTPDigestAuth, que é exigido pela API do Atlas para chaves de organização.
response = requests.post(
    url,
    json=data,
    auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY) 
)

if response.status_code == 201:
    print(f'[SUCESSO] IP {ip} adicionado à whitelist.')
elif response.status_code == 409:
    print(f'[AVISO] IP {ip} já está na whitelist.')
else:
    # Captura o erro e falha o build
    print(f'[ERRO] Falha ao adicionar IP: {response.status_code}')
    print(f'Detalhe da Resposta: {response.text}')
    sys.exit(1) # Força a falha do build