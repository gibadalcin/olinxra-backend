import requests
import os
import sys
import socket
import time  # <-- Adicione esta linha
from requests.auth import HTTPDigestAuth
from dotenv import load_dotenv

# Carrega variáveis do .env (Apenas para desenvolvimento local, DigitalOcean usa ENV vars diretas)
load_dotenv()

# Configurações do Atlas via ENV
ATLAS_PUBLIC_KEY = os.getenv('ATLAS_PUBLIC_KEY')
ATLAS_PRIVATE_KEY = os.getenv('ATLAS_PRIVATE_KEY')
ATLAS_PROJECT_ID = os.getenv('ATLAS_PROJECT_ID')
FORCE_IP = os.getenv('FORCE_PUBLIC_IP')  # Permite sobrescrever o IP manualmente
PROPAGATION_WAIT = int(os.getenv('ATLAS_PROPAGATION_WAIT', '3'))  # Tempo de espera para propagação (segundos)

if not ATLAS_PUBLIC_KEY or not ATLAS_PRIVATE_KEY or not ATLAS_PROJECT_ID:
    print("[ERRO] Variáveis ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY ou ATLAS_PROJECT_ID não encontradas!")
    sys.exit(1)

def get_public_ip():
    services = [
        'https://api.ipify.org',
        'https://ifconfig.me/ip',
        'https://ipinfo.io/ip'
    ]
    for service in services:
        try:
            resp = requests.get(service, timeout=5)
            resp.raise_for_status()
            ip = resp.text.strip()
            if ip:
                print(f'IP público detectado via {service}: {ip}')
                return ip
        except Exception as e:
            print(f'[AVISO] Falha ao obter IP via {service}: {e}')
    print('[ERRO] Não foi possível detectar o IP público.')
    sys.exit(1)

ip = FORCE_IP if FORCE_IP else get_public_ip()

# Exibe IP privado local para comparação
try:
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f'IP privado local: {local_ip}')
except Exception as e:
    print(f'[AVISO] Não foi possível obter o IP privado local: {e}')

# Adiciona o IP à whitelist do Atlas
url = f'https://cloud.mongodb.com/api/atlas/v1.0/groups/{ATLAS_PROJECT_ID}/accessList'
data = [
    {
        "ipAddress": ip,
        "comment": "IP do backend DigitalOcean (adicionado automaticamente)"
    }
]

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
    print(f'[ERRO] Falha ao adicionar IP: {response.status_code}')
    print(f'Detalhe da Resposta: {response.text}')
    sys.exit(1)

print(f'Aguardando {PROPAGATION_WAIT} segundos para propagação da whitelist no Atlas...')
time.sleep(PROPAGATION_WAIT)
print('Propagação concluída. Agora inicie o deploy do backend manualmente usando este IP liberado.')