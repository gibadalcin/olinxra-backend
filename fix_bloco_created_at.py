#!/usr/bin/env python3
"""Corrige blocos em `conteudos` sem o campo `created_at`.

Uso:
  python fix_bloco_created_at.py [--dry-run]

O que faz:
- Procura documentos na coleção `conteudos` com blocos onde `created_at` está ausente ou nulo.
- Para cada bloco faltante, preenche `created_at` com (em ordem de preferência):
  1) o valor de `updated_at` do documento (se for datetime),
  2) a hora atual (UTC).
- Por padrão faz um dry-run e só reporta; use --apply para gravar.

Importante: faça backup (snapshot) do banco antes de rodar com --apply em produção.
"""
import os
import argparse
import json
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId


def try_parse_datetime(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except Exception:
            try:
                return datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Não grava, apenas reporta')
    args = parser.parse_args()

    uri = os.getenv('MONGO_URI')
    if not uri:
        print('MONGO_URI não definida no ambiente. Saindo.')
        return

    client = MongoClient(uri)
    db_name = os.getenv('MONGO_DB_NAME', 'olinxra')
    db = client[db_name]
    conteudos = db['conteudos']

    query = {'blocos': {'$exists': True, '$ne': []}}
    cursor = conteudos.find(query)

    total_checked = 0
    total_updated = 0
    samples = []

    for doc in cursor:
        total_checked += 1
        blocos = doc.get('blocos') or []
        updated_at = try_parse_datetime(doc.get('updated_at'))
        modified = False
        new_blocos = []
        for b in blocos:
            if not isinstance(b, dict):
                new_blocos.append(b)
                continue
            created = b.get('created_at')
            parsed = try_parse_datetime(created)
            if parsed is None:
                # preencher com updated_at se disponível, senão agora
                fill = updated_at or datetime.utcnow()
                b['created_at'] = fill
                modified = True
            else:
                # garantir que seja datetime no objeto (não convertemos tipos no DB aqui)
                b['created_at'] = parsed
            new_blocos.append(b)

        if modified:
            total_updated += 1
            samples.append({'_id': str(doc.get('_id')), 'updated_blocks_count': sum(1 for b in blocos if not try_parse_datetime(b.get('created_at')) )})
            if not args.dry_run:
                try:
                    conteudos.update_one({'_id': doc['_id']}, {'$set': {'blocos': new_blocos}})
                except Exception as e:
                    print(f'Erro ao atualizar documento {doc.get("_id")}: {e}')

    result = {
        'total_docs_checked': total_checked,
        'total_docs_modified': total_updated,
        'dry_run': args.dry_run,
        'sample_modified': samples[:10]
    }

    print(json.dumps(result, default=str, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
