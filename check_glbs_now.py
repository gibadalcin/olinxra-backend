#!/usr/bin/env python3
"""Verifica se os GLBs estão nos items do carousel."""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Conectar ao MongoDB
mongo_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongo_uri)
db = client['olinxra']
collection = db['conteudos']

# Buscar g3
doc = collection.find_one({'marca': 'g3'})

if not doc:
    print("❌ Marca g3 não encontrada")
    exit(1)

print(f"✅ Marca g3 encontrada: {doc['_id']}\n")

blocos = doc.get('blocos', {}).get('blocos', [])
print(f"📦 Total de blocos: {len(blocos)}\n")

for i, bloco in enumerate(blocos):
    tipo = bloco.get('tipo', 'unknown')
    print(f"📋 Bloco {i}: {tipo}")
    
    if 'items' in bloco:
        items = bloco['items']
        print(f"   📊 Items: {len(items)}")
        for j, item in enumerate(items):
            nome = item.get('nome', 'sem nome')
            has_glb_url = 'glb_url' in item
            has_glb_signed = 'glb_signed_url' in item
            glb_url = item.get('glb_url', 'N/A')[:80] if has_glb_url else 'N/A'
            
            status = "✅" if has_glb_url else "❌"
            print(f"      {status} Item {j}: {nome}")
            if has_glb_url:
                print(f"         glb_url: {glb_url}...")
    print()
