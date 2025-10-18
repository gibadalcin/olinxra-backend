#!/usr/bin/env python3
"""Verifica o estado da migração de `marca_id` na coleção `conteudos`.

Uso:
  # no Powershell
  python check_migration.py

Requer: variável de ambiente MONGO_URI apontando para o banco (mesma que o backend usa).
"""
import os
import json
from pymongo import MongoClient


def human_doc_summary(doc):
    return {
        '_id': str(doc.get('_id')),
        'marca_id': str(doc.get('marca_id')),
        'nome_marca': doc.get('nome_marca'),
        'latitude': doc.get('latitude'),
        'longitude': doc.get('longitude'),
        'updated_at': str(doc.get('updated_at'))
    }


def main():
    uri = os.getenv('MONGO_URI')
    if not uri:
        print('Variável de ambiente MONGO_URI não definida. Configure e rode novamente.')
        return

    client = MongoClient(uri)
    # tenta usar o nome do DB das env (compatibilidade)
    db_name = os.getenv('MONGO_DB_NAME', 'olinxra')
    db = client[db_name]
    conteudos = db['conteudos']

    total = conteudos.count_documents({})
    count_string = conteudos.count_documents({'marca_id': {'$type': 'string'}})
    count_objectid = conteudos.count_documents({'marca_id': {'$type': 'objectId'}})
    count_missing = conteudos.count_documents({'marca_id': {'$exists': False}})

    print(json.dumps({
        'db': db_name,
        'total_conteudos': total,
        'marca_id_string_count': count_string,
        'marca_id_objectid_count': count_objectid,
        'marca_id_missing_count': count_missing
    }, indent=2))

    def print_samples(filter_q, title):
        print('\n---', title, '---')
        for doc in conteudos.find(filter_q).limit(5):
            print(json.dumps(human_doc_summary(doc), ensure_ascii=False))

    print_samples({'marca_id': {'$type': 'string'}}, 'Amostras: marca_id como string')
    print_samples({'marca_id': {'$type': 'objectId'}}, 'Amostras: marca_id como ObjectId')


if __name__ == '__main__':
    main()
