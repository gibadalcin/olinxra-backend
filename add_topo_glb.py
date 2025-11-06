#!/usr/bin/env python3
"""Adiciona glb_url para o bloco de imagem topo (bloco 0)."""
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
from datetime import datetime, UTC

load_dotenv()

mongo_uri = os.getenv('MONGO_URI')
print(f"🔌 Conectando ao MongoDB...")
client = MongoClient(mongo_uri)
db = client['olinxra']
collection = db['conteudos']
print(f"✅ Conectado! Database: olinxra, Collection: conteudos")

# Buscar documento do g3
doc_id = '69094441ea3606f1a22b24d0'
print(f"\n🔍 Buscando documento por _id: {doc_id}")
doc = collection.find_one({'_id': ObjectId(doc_id)})

if not doc:
    print("❌ Documento não encontrado!")
    exit(1)

print(f"✅ Documento encontrado: {doc['nome_marca']}")

# Nome do arquivo GLB no GCS (verifique qual é o nome correto)
# Você disse que tem no GCS, então preciso saber o nome exato
glb_filename = input("\n📝 Digite o nome do arquivo GLB da imagem topo no GCS (ex: topo.glb): ").strip()

if not glb_filename:
    print("❌ Nome do arquivo não pode ser vazio!")
    exit(1)

# Caminho completo no GCS
user_id = 'TR77xSOJrigOHfkoYQtx1iim6ok1'
glb_path = f"{user_id}/ra/models/{glb_filename}"
glb_url = f"gs://olinxra-conteudo/{glb_path}"  # Usar gs:// igual aos outros

print(f"\n🔧 Adicionando GLB ao bloco 0 (imagem topo):")
print(f"   glb_url: {glb_url}")
print(f"   glb_filename: {glb_filename}")

# Atualizar MongoDB - estrutura correta é blocos (array direto)
try:
    result = collection.update_one(
        {'_id': ObjectId(doc_id)},
        {
            '$set': {
                'blocos.0.glb_url': glb_url,
                'blocos.0.glb_filename': glb_path,
                'blocos.0.glb_source': 'retroactive_generation',
                'updated_at': datetime.now(UTC)
            }
        }
    )
    
    if result.modified_count > 0:
        print(f"\n✅ GLB adicionado com sucesso ao bloco 0!")
        print(f"   Documentos modificados: {result.modified_count}")
        print(f"\n🔄 Cache será invalidado automaticamente devido ao updated_at")
    else:
        print(f"\n⚠️ Nenhum documento foi modificado")
        
except Exception as e:
    print(f"\n❌ Erro ao atualizar: {e}")
    exit(1)

print("\n✅ Concluído!")
