# Análise do arquivo main.py - OlinxRA Backend

**Data da análise:** 06 de novembro de 2025  
**Arquivo analisado:** `main.py` (3005 linhas)

## 📋 Resumo Executivo

O arquivo `main.py` possui **~3000 linhas** e contém **27 endpoints principais**. Foram identificadas oportunidades significativas de otimização, incluindo código duplicado, endpoints redundantes e práticas que podem ser melhoradas.

---

## 🔍 Principais Descobertas

### 1. ⚠️ **CÓDIGO DUPLICADO - CRÍTICO**

#### 1.1 Função `_resize_if_needed` duplicada

**Localização:**
- Linha 1256 (dentro de `api_generate_glb_from_image`)
- Linha 2898 (dentro de `add_content_image`)

**Código duplicado:**
```python
def _resize_if_needed(src_path, max_dim):
    img = PILImage.open(src_path)
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        img = img.convert('RGB')
        img = img.resize(new_size, PILImage.LANCZOS)
        dst = src_path + '.resized.jpg'
        img.save(dst, format='JPEG', quality=90)
        return dst
    return src_path
```

**Impacto:** Dificulta manutenção e pode gerar inconsistências.

**Recomendação:** ✅ Criar função global única no topo do arquivo:
```python
def resize_image_if_needed(src_path: str, max_dim: int = 2048) -> str:
    """Redimensiona imagem se exceder dimensão máxima"""
    img = PILImage.open(src_path)
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        img = img.convert('RGB')
        img = img.resize(new_size, PILImage.LANCZOS)
        dst = src_path + '.resized.jpg'
        img.save(dst, format='JPEG', quality=90)
        return dst
    return src_path
```

---

### 2. 🔄 **ENDPOINTS REDUNDANTES/SIMILARES**

#### 2.1 Endpoints de busca de conteúdo

| Endpoint | Linha | Propósito | Status |
|----------|-------|-----------|--------|
| `POST /consulta-conteudo/` | 1656 | Busca conteúdo por marca + localização (sequencial) | ⚠️ Legacy |
| `POST /api/smart-content` | 1743 | Busca conteúdo otimizada (paralela) | ✅ Recomendado |
| `POST /api/conteudo` | 2022 | Cria/atualiza conteúdo | ✅ Manter |
| `GET /api/conteudo` | 1920 | Lista conteúdos por marca | ✅ Manter |
| `GET /api/conteudo-por-regiao` | 1982 | Busca conteúdo por região | ✅ Manter |

**Análise:**
- `/consulta-conteudo/` e `/api/smart-content` têm **propósito similar**, mas `/api/smart-content` é **~10x mais rápido** (2-3s vs 20s)
- `/api/smart-content` usa **lookups paralelos** e cache otimizado

**Recomendação:**
- ✅ **MANTER** `/api/smart-content` como endpoint principal
- ⚠️ **DEPRECAR** `/consulta-conteudo/` após migração dos clientes
- 📝 Adicionar header de depreciação: `Deprecation: true` em `/consulta-conteudo/`

#### 2.2 Endpoints de busca de logo

| Endpoint | Linha | Autenticação | Status |
|----------|-------|--------------|--------|
| `POST /search-logo/` | 927 | ❌ Não | ⚠️ Público |
| `POST /authenticated-search-logo/` | 931 | ✅ Sim | ✅ Recomendado |

**Problema:** Ambos usam a mesma função interna `_search_and_compare_logic()`

**Recomendação:**
- ✅ **MANTER** `/authenticated-search-logo/` como padrão
- ⚠️ **AVALIAR** necessidade de endpoint público `/search-logo/`
- 🔒 Se necessário público, adicionar rate limiting

---

### 3. 📦 **ESTRUTURA DE CACHE**

Foram identificados **2 caches globais** em memória:

```python
# Linha 46
geocode_cache = {}  # Cache de geocodificação reversa (limite: 1000 entradas)

# Linha 1649
consulta_cache = {}  # Cache de consultas de conteúdo (sem limite!)
```

**Problemas:**
1. ⚠️ `consulta_cache` **não tem limite de tamanho** → risco de vazamento de memória
2. ⚠️ Caches são **perdidos a cada restart** do servidor
3. ⚠️ **Não há TTL** (Time To Live) para invalidação

**Recomendação:**
- ✅ Implementar limite de tamanho para `consulta_cache` (ex: 1000 entradas)
- ✅ Adicionar TTL (ex: 1 hora) para evitar dados obsoletos
- 🚀 **Considerar Redis** para cache persistente e distribuído

---

### 4. 🎯 **LISTA COMPLETA DE ENDPOINTS**

#### 4.1 Endpoints de Conteúdo

| Método | Rota | Autenticação | Propósito |
|--------|------|--------------|-----------|
| GET | `/api/conteudo-signed-url` | ❌ | Gera signed URL para um arquivo |
| POST | `/api/conteudo-signed-urls` | ❌ | Gera signed URLs em batch |
| GET | `/api/default-totem-signed-url` | ❌ | URL do totem padrão (REMOVIDO) |
| POST | `/api/validate-button-block` | ✅ | Valida payload de bloco de botão |
| POST | `/api/generate-glb-from-image` | ❌ | Gera modelo GLB a partir de imagem |
| POST | `/consulta-conteudo/` | ❌ | Busca conteúdo (LEGACY) |
| POST | `/api/smart-content` | ❌ | Busca conteúdo (OTIMIZADO) ✅ |
| GET | `/api/marcas` | ❌ | Lista marcas disponíveis |
| GET | `/api/conteudo` | ✅ | Lista conteúdos por marca |
| GET | `/api/conteudo-por-regiao` | ❌ | Busca conteúdo por região |
| POST | `/api/conteudo` | ✅ | Cria/atualiza conteúdo |
| POST | `/add-content-image/` | ✅ | Upload de imagem para conteúdo |

#### 4.2 Endpoints de Logo (Reconhecimento de Marca)

| Método | Rota | Autenticação | Propósito |
|--------|------|--------------|-----------|
| POST | `/search-logo/` | ❌ | Busca logo por imagem (público) |
| POST | `/authenticated-search-logo/` | ✅ | Busca logo por imagem (auth) |
| POST | `/add-logo/` | ✅ | Adiciona nova logo ao banco |
| DELETE | `/delete-logo/` | ✅ | Remove logo do banco |
| GET | `/images` | ❌ | Lista imagens (DEPRECATED?) |

#### 4.3 Endpoints de Upload/Assets

| Método | Rota | Autenticação | Propósito |
|--------|------|--------------|-----------|
| POST | `/upload/cancel` | ✅ | Cancela upload e remove asset |
| POST | `/admin/cleanup-uploaded-assets` | ✅ | Limpa assets órfãos |

#### 4.4 Endpoints de Administração

| Método | Rota | Autenticação | Propósito |
|--------|------|--------------|-----------|
| GET | `/admin/list` | ✅ | Lista usuários admin |
| POST | `/admin/create` | ✅ | Cria novo admin |
| POST | `/admin/delete` | ✅ | Remove admin |
| POST | `/admin/process-pending-deletes` | ✅ | Processa deleções pendentes |

#### 4.5 Endpoints de Debug/Utilities

| Método | Rota | Autenticação | Propósito |
|--------|------|--------------|-----------|
| GET | `/debug/user` | ✅ | Mostra info do usuário autenticado |
| GET | `/debug/logos` | ❌ | Lista logos no banco |
| POST | `/debug/inspect-request/` | ❌ | Inspeciona payload recebido |
| GET | `/api/reverse-geocode` | ❌ | Geocodificação reversa (lat/lon → endereço) |

---

### 5. 🔐 **ANÁLISE DE SEGURANÇA**

#### Endpoints sem autenticação que manipulam recursos:

1. ⚠️ `POST /search-logo/` - Processamento de imagem via CLIP
2. ⚠️ `POST /api/generate-glb-from-image` - Geração de GLB (computacionalmente caro)
3. ⚠️ `POST /debug/inspect-request/` - Endpoint de debug em produção?

**Recomendações:**
- 🔒 Adicionar rate limiting em endpoints públicos
- 🔒 Considerar autenticação ou API key para `/api/generate-glb-from-image`
- ⚠️ Remover endpoints de debug em produção ou proteger com autenticação

---

### 6. 📊 **WORKERS/BACKGROUND TASKS**

```python
# Linha 50-106: uploaded_assets_cleanup_worker
```

**Propósito:** Limpa assets órfãos (não vinculados a conteúdos) com TTL de 7 dias

**Status:** ✅ Implementação correta

**Configuração:**
- Intervalo: 24 horas (padrão)
- TTL: 7 dias (padrão)
- Ações: Deleta arquivos do GCS + remove do MongoDB

---

### 7. 🚀 **FUNÇÕES OTIMIZADAS IDENTIFICADAS**

#### 7.1 attach_signed_urls_to_blocos vs attach_signed_urls_to_blocos_fast

| Função | Linha | Otimização | Uso |
|--------|-------|------------|-----|
| `attach_signed_urls_to_blocos` | 579 | Padrão (verifica existência) | Legacy |
| `attach_signed_urls_to_blocos_fast` | 666 | Skip exists check + TTL 7 dias | ✅ Recomendado |

**Impacto:** Reduz tempo de geração de signed URLs em ~40-60%

**Recomendação:** Migrar todos os usos para versão `_fast`

---

## 🎯 **RECOMENDAÇÕES PRIORITÁRIAS**

### Prioridade ALTA ⚡

1. **Remover duplicação de `_resize_if_needed`**
   - Criar função global única
   - Atualizar chamadas nas linhas 1256 e 2898

2. **Adicionar limite ao `consulta_cache`**
   ```python
   consulta_cache = {}
   CONSULTA_CACHE_MAX_SIZE = 1000
   
   def add_to_cache(key, value):
       if len(consulta_cache) >= CONSULTA_CACHE_MAX_SIZE:
           # Remove item mais antigo (FIFO)
           consulta_cache.pop(next(iter(consulta_cache)))
       consulta_cache[key] = value
   ```

3. **Deprecar `/consulta-conteudo/`**
   - Adicionar header de depreciação
   - Atualizar documentação para usar `/api/smart-content`

### Prioridade MÉDIA 📊

4. **Consolidar endpoints de busca de logo**
   - Avaliar necessidade de endpoint público
   - Considerar rate limiting

5. **Migrar para `attach_signed_urls_to_blocos_fast`**
   - Substituir chamadas da versão lenta
   - Verificar impacto em clientes existentes

6. **Adicionar rate limiting**
   - Endpoints públicos de processamento pesado
   - Usar middleware (ex: slowapi)

### Prioridade BAIXA 📝

7. **Remover/Proteger endpoints de debug**
   - `/debug/inspect-request/`
   - Adicionar flag de ambiente (DEBUG=true)

8. **Refatorar cache para Redis**
   - Cache persistente entre restarts
   - TTL automático
   - Cache distribuído (múltiplas instâncias)

---

## 📈 **MÉTRICAS DO ARQUIVO**

- **Total de linhas:** 3005
- **Total de endpoints:** 27
- **Total de funções:** ~50+
- **Endpoints autenticados:** 15 (55%)
- **Endpoints públicos:** 12 (45%)
- **Código duplicado identificado:** 2 funções
- **Caches globais:** 2

---

## ✅ **PONTOS POSITIVOS**

1. ✅ Uso adequado de async/await
2. ✅ Background worker para limpeza de assets
3. ✅ Implementação de cache (geocode e consulta)
4. ✅ Signed URLs com expiração configurável
5. ✅ Separação de lógica de geração de GLB
6. ✅ Endpoint otimizado `/api/smart-content` com lookups paralelos
7. ✅ Versionamento de funções (normal vs fast)

---

## 🔧 **PRÓXIMOS PASSOS**

1. [ ] Criar branch para refatoração
2. [ ] Implementar função global `resize_image_if_needed`
3. [ ] Adicionar limite ao `consulta_cache`
4. [ ] Adicionar testes unitários para funções críticas
5. [ ] Documentar endpoints (OpenAPI/Swagger)
6. [ ] Implementar rate limiting
7. [ ] Migrar cache para Redis (se necessário)
8. [ ] Criar ADR (Architecture Decision Record) para decisões importantes

---

## 📚 **DOCUMENTAÇÃO RELACIONADA**

- `olinxra-adminui/ENDPOINTS.md` - Documentação de endpoints do admin
- `docs/CAMADAS-DE-ACESSO.md` - Arquitetura de camadas de acesso
- `docs/SINCRONIZACAO-DELECAO-GLB.md` - Sincronização de GLBs

---

**Análise gerada por:** GitHub Copilot  
**Revisar com equipe antes de implementar mudanças**
