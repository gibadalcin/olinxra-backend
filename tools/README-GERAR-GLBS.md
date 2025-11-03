# 🔄 Gerar GLBs para Imagens Existentes

## Resumo

Você **NÃO precisa** deletar e re-fazer upload das imagens! 🎉

Este script processa todos os conteúdos existentes no MongoDB e gera GLBs automaticamente para imagens que ainda não têm.

---

## 📋 Pré-requisitos

1. **Backend rodando**: `python run.py` em `olinxra-backend/`
2. **MongoDB acessível**: variável `MONGO_URI` no `.env`
3. **Dependências instaladas**: `motor`, `httpx`, `python-dotenv`

---

## 🚀 Como usar

### 1. Dry-run (simular sem modificar)

```bash
cd olinxra-backend
python tools/generate_glbs_from_existing_images.py --dry-run
```

**O que faz:**
- ✅ Lista todos os documentos de conteúdo
- ✅ Mostra quais imagens receberiam GLBs
- ✅ Não modifica nada no MongoDB
- ✅ Não gera GLBs de verdade

**Saída esperada:**
```
📊 Total de documentos encontrados: 15
📄 Processando documento: 67890... (marca: olinx, blocos: 5)
  🔍 [DRY-RUN] Seria gerado GLB para totem_header.jpg
  ⏭️  Bloco 1 já tem GLB: gs://bucket/ra/models/logo.glb
  🎠 Bloco 2: carousel com 3 imagens
    🔍 [DRY-RUN] Seria gerado GLB para carousel_1.jpg
    🔍 [DRY-RUN] Seria gerado GLB para carousel_2.jpg
...
📊 RESUMO FINAL
Imagens processadas: 48
GLBs gerados: 0 (dry-run)
GLBs já existentes: 5
```

### 2. Gerar GLBs de verdade

```bash
python tools/generate_glbs_from_existing_images.py
```

**O que faz:**
- ✅ Busca todos os blocos de tipo `imagem` e `carousel`
- ✅ Para cada imagem sem `glb_url`:
  - Gera signed URL da imagem original
  - Chama `/api/generate-glb-from-image`
  - Adiciona `glb_url` e `glb_signed_url` ao bloco
- ✅ Atualiza documento no MongoDB
- ✅ Pula imagens que já têm GLB

**Saída esperada:**
```
📄 Processando documento: 67890... (marca: olinx, blocos: 5)
  🔄 Bloco 0: gerando GLB para totem_header.jpg
  ✅ GLB gerado: gs://bucket/{uid}/ra/models/totem_header.glb
  ✅ GLB adicionado ao bloco 0
  ⏭️  Bloco 1 já tem GLB (pulado)
  🎠 Bloco 2: carousel com 3 imagens
    🔄 Imagem 0: gerando GLB para carousel_1.jpg
    ✅ GLB adicionado à imagem 0
    ...
✅ Documento 67890 atualizado no MongoDB

📊 RESUMO FINAL
Documentos processados: 15
Imagens processadas: 48
GLBs gerados: 43
GLBs já existentes: 5
Erros: 0
```

### 3. Processar apenas uma marca específica

```bash
python tools/generate_glbs_from_existing_images.py --marca olinx
```

**Útil para:**
- Testar com uma marca primeiro
- Processar marcas em lotes

### 4. Limitar número de documentos (testes)

```bash
python tools/generate_glbs_from_existing_images.py --limit 5
```

**Útil para:**
- Testar o script com poucos documentos
- Verificar se está funcionando antes de processar tudo

### 5. Combinar opções

```bash
# Dry-run de 10 documentos da marca "olinx"
python tools/generate_glbs_from_existing_images.py --dry-run --marca olinx --limit 10
```

---

## 📊 O que o script faz

### Para cada documento de conteúdo:

1. **Busca blocos de imagem:**
   - Tipo `imagem` (header, card, etc.)
   - Tipo `carousel` (array de imagens)

2. **Verifica se GLB já existe:**
   - Se `bloco.glb_url` existe → pula
   - Se não existe → gera GLB

3. **Gera GLB:**
   - Gera signed URL da imagem original (via `/api/conteudo-signed-url`)
   - Chama `/api/generate-glb-from-image` com a signed URL
   - Backend gera GLB e salva no GCS
   - Backend retorna `glb_signed_url` e `gs_url`

4. **Atualiza MongoDB:**
   - Adiciona `glb_url` ao bloco
   - Adiciona `glb_signed_url` ao bloco
   - Adiciona `glb_generated_at` (timestamp)
   - Atualiza `glb_last_updated` no documento

---

## 🔍 Estrutura do documento atualizado

**Antes:**
```json
{
  "_id": "...",
  "nome_marca": "olinx",
  "owner_uid": "user123",
  "blocos": [
    {
      "tipo": "imagem",
      "url": "gs://bucket/image.jpg",
      "signed_url": "https://..."
    }
  ]
}
```

**Depois:**
```json
{
  "_id": "...",
  "nome_marca": "olinx",
  "owner_uid": "user123",
  "blocos": [
    {
      "tipo": "imagem",
      "url": "gs://bucket/image.jpg",
      "signed_url": "https://...",
      "glb_url": "gs://bucket/user123/ra/models/image.glb",         // ← NOVO
      "glb_signed_url": "https://storage.googleapis.com/...",        // ← NOVO
      "glb_generated_at": "2025-11-03T..."                           // ← NOVO
    }
  ],
  "glb_last_updated": "2025-11-03T..."                               // ← NOVO
}
```

---

## ⚡ Performance

### Tempo estimado por imagem:
- Pequena (< 500KB): ~2-3s
- Média (1-2MB): ~3-5s
- Grande (> 5MB): ~5-8s

### Tempo total estimado:
- 10 imagens: ~30-50s
- 50 imagens: ~2-5min
- 100 imagens: ~5-10min

**Cache:**
- Se GLB já foi gerado antes (mesmo hash), backend retorna imediatamente (< 1s)
- GLBs existentes são pulados automaticamente

---

## 🐛 Troubleshooting

### Erro: "Backend não está respondendo"
```
❌ Exceção ao gerar GLB: Connection refused
```

**Solução:**
```bash
# Verificar se backend está rodando
cd olinxra-backend
python run.py
```

### Erro: "MONGO_URI não configurado"
```
RuntimeError: MONGO_URI não configurado no .env
```

**Solução:**
```bash
# Verificar .env
cat .env | grep MONGO_URI

# Ou adicionar:
echo 'MONGO_URI="mongodb+srv://..."' >> .env
```

### Erro: "Failed to download image"
```
❌ Erro ao gerar GLB: 400 - Failed to download image
```

**Causa:** Signed URL da imagem original expirou

**Solução:**
- Script gera nova signed URL automaticamente
- Se persistir, verificar permissões GCS

### Algumas imagens falharam
```
📊 RESUMO FINAL
Erros: 5
```

**Verificar:**
1. Logs para ver qual imagem falhou
2. Verificar se imagem existe no GCS
3. Verificar tamanho da imagem (> 5MB pode falhar)
4. Re-executar script (vai processar apenas as que falharam)

---

## 🧪 Validação

### 1. Verificar no MongoDB
```javascript
// Contar documentos com GLBs
db.conteudos.countDocuments({
  "blocos.glb_url": { $exists: true }
})

// Ver exemplo
db.conteudos.findOne({
  "blocos.glb_url": { $exists: true }
}, {
  "blocos.$": 1
})
```

### 2. Verificar no GCS
```bash
# Listar GLBs gerados
gsutil ls gs://olinxra-conteudo/{seu_uid}/ra/models/

# Ver tamanho total
gsutil du -sh gs://olinxra-conteudo/{seu_uid}/ra/models/
```

### 3. Testar GLB no viewer
1. Pegar `glb_signed_url` do MongoDB
2. Abrir: https://gltf-viewer.donmccurdy.com/
3. Colar URL → verificar se modelo aparece

---

## 📝 Logs importantes

### Sucesso:
```
✅ GLB gerado: gs://bucket/user123/ra/models/image.glb (cached: False)
✅ GLB adicionado ao bloco 0
✅ Documento 67890 atualizado no MongoDB
```

### Cache hit (GLB já existia):
```
✅ GLB gerado: gs://bucket/user123/ra/models/image.glb (cached: True)
```

### Pulado (já tem glb_url):
```
⏭️  Bloco 1 já tem GLB: gs://bucket/ra/models/logo.glb
```

### Erro:
```
❌ Erro ao gerar GLB: 500 - Internal Server Error
❌ Erro ao atualizar documento 67890: ...
```

---

## ✅ Checklist pós-execução

- [ ] Script executado sem erros
- [ ] MongoDB atualizado (verificar `glb_url` nos documentos)
- [ ] GLBs acessíveis no GCS
- [ ] GLBs visualizados corretamente no gltf-viewer
- [ ] App mobile pode carregar GLBs (próxima fase)

---

## 🎯 Próximos passos

Após gerar GLBs para imagens existentes:

1. **✅ Validar no MongoDB** - campos `glb_url` presentes
2. **✅ Testar no App Mobile** - FASE 3 (extrair GLBs em `ar-view.tsx`)
3. **✅ Implementar navegação** - FASE 4 (controles AR)

---

**Última atualização:** 03/11/2025  
**Script:** `tools/generate_glbs_from_existing_images.py`
