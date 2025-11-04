# Fix: Incluir GLBs nos Itens do Carousel

## Problema Identificado

Atualmente, quando o backend retorna o conteúdo de uma marca, os **itens do carousel NÃO incluem os GLBs** gerados automaticamente.

### Estrutura Atual (INCORRETA)

```json
{
  "blocos": [
    {
      "tipo": "Carousel 1",
      "items": [
        {
          "subtipo": "card",
          "url": "gs://...",
          "signed_url": "https://storage.googleapis.com/...",
          "nome": "juninho-card.png"
          // ❌ FALTA: glb_signed_url ou glb_url
        }
      ]
    }
  ]
}
```

### Estrutura Esperada (CORRETA)

```json
{
  "blocos": [
    {
      "tipo": "Carousel 1",
      "items": [
        {
          "subtipo": "card",
          "url": "gs://...",
          "signed_url": "https://storage.googleapis.com/...",
          "nome": "juninho-card.png",
          "glb_url": "gs://olinxra-conteudo/.../juninho-card.glb",           // ✅ ADICIONAR
          "glb_signed_url": "https://storage.googleapis.com/.../juninho-card.glb?X-Goog..." // ✅ ADICIONAR
        }
      ]
    }
  ]
}
```

## Evidências dos Logs

```
LOG  [ARView] 🔍 Item 0 do bloco 1: {
  "temGlbSignedUrl": false,  // ❌ Deveria ser true
  "temGlbUrl": false,        // ❌ Deveria ser true
  "temSignedUrl": true,
  "temUrl": true
}
LOG  [ARView] ❌ Item 0 do bloco 1 NÃO tem GLB
```

## Solução Necessária

### 1. Endpoint a Modificar

Provavelmente é o endpoint que busca conteúdo por marca. Pode ser:
- `POST /api/compare-logo` (resposta quando logo é reconhecida)
- `GET /api/conteudo/:marca`
- Ou outro endpoint que retorna os blocos

### 2. Lógica a Implementar

Para **cada item** dentro de `blocos[x].items[]`:

1. **Verificar se GLB já existe**:
   ```python
   # Exemplo com o item do carousel
   item_image_url = item.get('url') or item.get('signed_url')
   
   # Gerar nome do GLB baseado na imagem
   # Exemplo: "juninho-card.png" -> "juninho-card.glb"
   glb_filename = item_image_url.replace('.png', '.glb').replace('.jpg', '.glb').replace('.jpeg', '.glb')
   
   # Verificar se GLB existe no Cloud Storage
   if glb_exists_in_storage(glb_filename):
       item['glb_url'] = f"gs://olinxra-conteudo/{glb_filename}"
       item['glb_signed_url'] = generate_signed_url(glb_filename)
   ```

2. **Se GLB não existir** (opcional):
   - Pode gerar GLB sob demanda ou
   - Deixar vazio (app gerará quando usuário clicar "Ver em RA")

### 3. Onde Adicionar a Lógica

Provavelmente em `main.py` ou `firebase_utils.py`, na função que monta a resposta dos blocos:

```python
def get_conteudo_marca(marca_id):
    # ... código existente que busca blocos ...
    
    for bloco in blocos:
        if bloco.get('tipo', '').lower().startswith('carousel'):
            items = bloco.get('items', [])
            for item in items:
                # ✅ ADICIONAR AQUI: Buscar GLB para este item
                item_glb_url = find_or_generate_glb_for_item(item)
                if item_glb_url:
                    item['glb_url'] = item_glb_url['gs']
                    item['glb_signed_url'] = item_glb_url['signed']
    
    return blocos
```

## Verificação

Após a modificação, verificar que a resposta do backend inclui:

```json
{
  "blocos": [
    {
      "tipo": "Carousel 1",
      "items": [
        {
          "nome": "juninho-card.png",
          "signed_url": "https://...",
          "glb_url": "gs://...",           // ✅ DEVE EXISTIR
          "glb_signed_url": "https://..."  // ✅ DEVE EXISTIR
        },
        {
          "nome": "jean-card.png",
          "signed_url": "https://...",
          "glb_url": "gs://...",           // ✅ DEVE EXISTIR
          "glb_signed_url": "https://..."  // ✅ DEVE EXISTIR
        }
      ]
    }
  ]
}
```

## Impacto no Frontend

Após a correção, o app vai:
1. ✅ Detectar GLBs nos itens do carousel
2. ✅ Mostrar controles de navegação "◀ 1/4 ▶"
3. ✅ Permitir navegar entre os modelos 3D
4. ✅ Abrir AR direto (sem precisar gerar GLB sob demanda)

## Arquivos Backend para Verificar

- [ ] `main.py` - Endpoints de conteúdo
- [ ] `firebase_utils.py` - Funções de busca de blocos
- [ ] `schemas.py` - Schema de resposta dos blocos
- [ ] `gcs_utils.py` - Funções de Cloud Storage (verificar se GLB existe)

## Próximos Passos

1. ✅ Criar este documento
2. ⏳ Modificar backend para incluir GLBs nos itens do carousel
3. ⏳ Testar resposta do backend (verificar JSON)
4. ⏳ Testar no app (verificar se controles de navegação aparecem)
5. ⏳ Commit das alterações

---

**Data**: 2025-11-04  
**Relacionado**: Issue de múltiplos GLBs em carousel  
**Prioridade**: ALTA (bloqueando funcionalidade de navegação entre modelos)
