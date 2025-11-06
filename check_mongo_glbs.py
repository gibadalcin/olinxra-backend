#!/usr/bin/env python3
"""Verifica se glb_url está no MongoDB."""
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import json

load_dotenv()

mongo_uri = os.getenv('MONGO_URI')  # Correção: é MONGO_URI, não MONGODB_URI
print(f"🔌 Conectando ao MongoDB...")
client = MongoClient(mongo_uri)
db = client['olinxra']
collection = db['conteudos']
print(f"✅ Conectado! Database: olinxra, Collection: conteudos")

# Buscar documento do g3 pelo ID correto
doc_id = '69094441ea3606f1a22b24d0'
print(f"\n🔍 Buscando documento por _id: {doc_id}")
try:
    doc = collection.find_one({'_id': ObjectId(doc_id)})
    if doc:
        print(f"✅ Encontrado por _id!")
except Exception as e:
    print(f"❌ Erro ao buscar por _id: {e}")
    doc = None

if not doc:
    print(f"\n⚠️ Tentando buscar por nome_marca='g3'...")
    doc = collection.find_one({'nome_marca': 'g3'})
    if doc:
        print(f"✅ Encontrado por nome_marca!")

if not doc:
    print("\n❌ Documento g3 não encontrado!")
    print(f"🔍 Total de documentos: {collection.count_documents({})}")
    print(f"\n🔍 Primeiros 3 documentos:")
    for d in collection.find().limit(3):
        print(f"   - _id: {d.get('_id')} | nome_marca: '{d.get('nome_marca', 'N/A')}'")
    exit(1)

print(f"✅ Documento g3 encontrado: {doc['_id']}")
print(f"   nome_marca: {doc.get('nome_marca')}")
print(f"   updated_at: {doc.get('updated_at')}\n")

blocos = doc.get('blocos', [])
if isinstance(blocos, dict):
    blocos = blocos.get('blocos', [])

for i, bloco in enumerate(blocos):
    if 'items' in bloco:
        print(f"📦 Bloco {i} ({bloco.get('tipo')}): {len(bloco['items'])} items")
        for j, item in enumerate(bloco['items']):
            print(f"\n   Item {j}: {item.get('nome', 'sem nome')}")
            print(f"      Keys: {list(item.keys())}")
            
            if 'glb_url' in item:
                print(f"      ✅ glb_url: {item['glb_url']}")
            else:
                print(f"      ❌ SEM glb_url")
            
            if 'glb_filename' in item:
                print(f"      ✅ glb_filename: {item['glb_filename']}")
            else:
                print(f"      ❌ SEM glb_filename")
