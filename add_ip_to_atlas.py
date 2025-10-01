import requests
import os
from requests.auth import HTTPDigestAuth
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

# Configurações do Atlas via .env
ATLAS_PUBLIC_KEY = os.getenv('ATLAS_PUBLIC_KEY')
ATLAS_PRIVATE_KEY = os.getenv('ATLAS_PRIVATE_KEY')
ATLAS_PROJECT_ID = os.getenv('ATLAS_PROJECT_ID')

if not ATLAS_PUBLIC_KEY or not ATLAS_PRIVATE_KEY or not ATLAS_PROJECT_ID:
    print("[ERRO] Variáveis ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY ou ATLAS_PROJECT_ID não encontradas no .env!")
    exit(1)

# 1. Descobre o IP público atual
ip = requests.get('https://api.ipify.org').text
print(f'IP público detectado: {ip}')

# 2. Adiciona o IP à whitelist do Atlas
url = f'https://cloud.mongodb.com/api/atlas/v1.0/groups/{ATLAS_PROJECT_ID}/accessList'
data = {
    "ipAddress": ip,
    "comment": "IP do backend DigitalOcean (adicionado automaticamente)"
}

response = requests.post(url, json=data, auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY))

if response.status_code == 201:
    print(f'IP {ip} adicionado à whitelist com sucesso!')
elif response.status_code == 409:
    print(f'IP {ip} já está na whitelist.')
else:
    print(f'[ERRO] Falha ao adicionar IP: {response.status_code} - {response.text}')
