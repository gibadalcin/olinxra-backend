#!/usr/bin/env python3
"""
Atualiza o campo updated_at de um conteúdo para invalidar cache no app.
Útil após adicionar GLBs retroativamente.
"""
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, UTC
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['olinxra']
collection = db['conteudos']

# ID do conteúdo g3
doc_id = '69094441ea3606f1a22b24d0'

print(f"🔄 Atualizando timestamp do conteúdo g3...")
result = collection.update_one(
    {'_id': ObjectId(doc_id)},
    {'$set': {'updated_at': datetime.now(UTC)}}
)

if result.modified_count > 0:
    print(f"✅ Timestamp atualizado!")
    print(f"   Isso vai invalidar o cache no app na próxima busca")
    
    # Verificar novo timestamp
    doc = collection.find_one({'_id': ObjectId(doc_id)}, {'updated_at': 1})
    print(f"   Novo updated_at: {doc['updated_at']}")
else:
    print(f"❌ Nenhum documento foi atualizado")
