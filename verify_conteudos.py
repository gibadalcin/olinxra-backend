#!/usr/bin/env python3
"""Verificação de consistência dos documentos em `conteudos`.

Verifica:
- `location` existe e é GeoJSON Point com 2 coordenadas numéricas
- `updated_at` não é string (espera-se datetime)
- Para cada `bloco` em `blocos`, `created_at` não é string (espera-se datetime)

Uso:
  python verify_conteudos.py

Requer MONGO_URI no ambiente.
"""
import os
import json
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId


def summary_doc(doc):
    return {
        '_id': str(doc.get('_id')),
        'marca_id': str(doc.get('marca_id')) if doc.get('marca_id') is not None else None,
        'nome_marca': doc.get('nome_marca'),
        'latitude': doc.get('latitude'),
        'longitude': doc.get('longitude'),
        'location': doc.get('location') is not None,
        'updated_at': type(doc.get('updated_at')).__name__ if 'updated_at' in doc else 'missing',
        'blocos_len': len(doc.get('blocos') or [])
    }


def is_geo_point(location):
    try:
        if not isinstance(location, dict):
            return False
        if location.get('type') != 'Point':
            return False
        coords = location.get('coordinates')
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            return False
        lon, lat = coords[0], coords[1]
        return isinstance(lon, (int, float)) and isinstance(lat, (int, float))
    except Exception:
        return False


def main():
    uri = os.getenv('MONGO_URI')
    if not uri:
        print('Variável de ambiente MONGO_URI não definida. Configure e rode novamente.')
        return

    client = MongoClient(uri)
    db_name = os.getenv('MONGO_DB_NAME', 'olinxra')
    db = client[db_name]
    conteudos = db['conteudos']

    total = conteudos.count_documents({})

    stats = {
        'total': total,
        'missing_location': 0,
        'invalid_location': 0,
        'updated_at_string': 0,
        'updated_at_missing': 0,
        'blocos_total': 0,
        'blocos_missing_created_at': 0,
        'blocos_created_at_string': 0,
    }

    problematic = []

    for doc in conteudos.find({}):
        doc_problems = []
        loc = doc.get('location')
        if loc is None:
            stats['missing_location'] += 1
            doc_problems.append('missing_location')
        else:
            if not is_geo_point(loc):
                stats['invalid_location'] += 1
                doc_problems.append('invalid_location')

        updated = doc.get('updated_at')
        if updated is None:
            stats['updated_at_missing'] += 1
            doc_problems.append('updated_at_missing')
        else:
            if not isinstance(updated, datetime):
                stats['updated_at_string'] += 1
                doc_problems.append('updated_at_not_datetime')

        blocos = doc.get('blocos') or []
        stats['blocos_total'] += len(blocos)
        for idx, b in enumerate(blocos):
            created = b.get('created_at')
            if created is None:
                stats['blocos_missing_created_at'] += 1
                doc_problems.append(f'bloco_{idx}_missing_created_at')
            else:
                if not isinstance(created, datetime):
                    stats['blocos_created_at_string'] += 1
                    doc_problems.append(f'bloco_{idx}_created_at_not_datetime')

        if doc_problems:
            problem_summary = summary_doc(doc)
            problem_summary['problems'] = doc_problems
            problematic.append(problem_summary)

    output = {
        'db': db_name,
        'stats': stats,
        'problematic_sample_count': len(problematic),
        'problematic_samples': problematic[:10]
    }

    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
