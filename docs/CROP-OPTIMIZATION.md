# 🎯 Otimização de Crop para Reconhecimento de Logos

## Problema Identificado

**Data:** 30/11/2025  
**Logs analisados:** Digital Ocean backend (01/12/2025 00:53-00:59)

### Sintomas

1. **Crop adaptativo removendo 10% inferior** das imagens antes do reconhecimento
2. **Logos posicionados na parte inferior** sendo cortados
3. **Scores de reconhecimento muito baixos:**
   - Combined scores: 0.368, 0.406, 0.447 (threshold: 0.72)
   - Margens entre candidatos: 0.0117, 0.0337, 0.0458 (< 0.05 mínimo)
4. **Falhas de reconhecimento** de logos conhecidos (Lenovo, Logitech, BossAuto)

### Logs de Exemplo

```
Dec 01 00:53:42  INFO [search_logo] removed bottom 10.0% (66px) from image
Dec 01 00:53:42  INFO [search_logo] adaptive crop used bbox=(0, 0, 350, 600)
Dec 01 00:53:42  INFO [search_logo] margin too small: d2(0.6484) - d1(0.6367) = 0.0117 < 0.05
```

## Solução Aplicada

### Alterações no `.env`

Adicionadas as seguintes variáveis de configuração:

```env
# Desabilita remoção da parte inferior (evita cortar logos na base da imagem)
SEARCH_REMOVE_BOTTOM_PCT=0.0

# Ajusta parâmetros do crop adaptativo para ser mais conservador
SEARCH_ADAPTIVE_SEED_RATIO=0.30          # aumentado de 0.20 (crop inicial maior)
SEARCH_ADAPTIVE_STEP_RATIO=0.10          # aumentado de 0.05 (passos maiores)
SEARCH_ADAPTIVE_MAX_EXPAND_RATIO=0.70    # aumentado de 0.50 (permite crop maior)
SEARCH_ADAPTIVE_EDGE_TH=0.02             # aumentado de 0.01 (menos sensível a bordas)
SEARCH_ADAPTIVE_VAR_RATIO_MIN=0.30       # reduzido de 0.40 (aceita menos variância)
SEARCH_ADAPTIVE_MIN_AREA_RATIO=0.05      # aumentado de 0.01 (área mínima maior)
```

### Raciocínio

1. **`SEARCH_REMOVE_BOTTOM_PCT=0.0`**: Desabilita completamente a remoção da faixa inferior, garantindo que logos posicionados na base das imagens sejam incluídos no crop.

2. **`SEARCH_ADAPTIVE_SEED_RATIO=0.30`**: Inicia com um crop maior (30% do menor lado ao invés de 20%), capturando mais contexto visual desde o início.

3. **`SEARCH_ADAPTIVE_STEP_RATIO=0.10`**: Aumenta o incremento de expansão do crop (10% ao invés de 5%), reduzindo o número de iterações e acelerando a convergência.

4. **`SEARCH_ADAPTIVE_MAX_EXPAND_RATIO=0.70`**: Permite que o crop alcance até 70% da imagem (antes 50%), capturando logos maiores ou em posições não-centrais.

5. **`SEARCH_ADAPTIVE_VAR_RATIO_MIN=0.30`**: Reduz o requisito de variância mínima (aceita regiões com menos contraste), útil para logos com cores suaves ou fundos uniformes.

6. **`SEARCH_ADAPTIVE_MIN_AREA_RATIO=0.05`**: Aumenta a área mínima aceitável do crop (5% ao invés de 1%), evitando crops muito pequenos que perdem contexto.

## Validação e Monitoramento

### Métricas a Observar

1. **Taxa de reconhecimento:**
   - Antes: ~30% de falsos negativos em logos conhecidos
   - Meta: <10% de falsos negativos

2. **Scores de similaridade:**
   - Embedding (d1): deve ficar < 0.65
   - Combined score: deve ficar > 0.72
   - Margin (d2-d1): deve ficar > 0.05

3. **Bbox do crop adaptativo:**
   - Verificar que não está cortando logos visíveis
   - Log: `[search_logo] adaptive crop used bbox=(...)`

### Como Testar

1. **Teste manual via AdminUI:**
   - Fazer upload de imagens com logos em diferentes posições (topo, centro, base)
   - Verificar logs do backend para bbox e scores
   - Confirmar reconhecimento bem-sucedido

2. **Teste via app mobile:**
   - Capturar fotos de logos reais em diferentes ângulos
   - Observar taxa de reconhecimento
   - Verificar tempo de resposta (deve manter < 200ms)

3. **Monitoramento de logs (Digital Ocean):**
```bash
# Filtrar logs de reconhecimento
doctl apps logs <app-id> --type run | grep "search_logo"

# Contar falhas de margin
doctl apps logs <app-id> --type run | grep "margin too small" | wc -l

# Verificar scores médios
doctl apps logs <app-id> --type run | grep "combined score"
```

### Rollback (se necessário)

Se a mudança causar regressão (aumento de falsos positivos ou lentidão):

```env
# Voltar aos valores anteriores
SEARCH_REMOVE_BOTTOM_PCT=0.10
SEARCH_ADAPTIVE_SEED_RATIO=0.20
SEARCH_ADAPTIVE_STEP_RATIO=0.05
SEARCH_ADAPTIVE_MAX_EXPAND_RATIO=0.50
SEARCH_ADAPTIVE_EDGE_TH=0.01
SEARCH_ADAPTIVE_VAR_RATIO_MIN=0.40
SEARCH_ADAPTIVE_MIN_AREA_RATIO=0.01
```

## Próximos Passos

1. **Deploy da alteração no Digital Ocean:**
   - Push do `.env` atualizado
   - Restart do app para aplicar as novas variáveis

2. **Testes em produção:**
   - Monitorar logs por 24-48h
   - Coletar métricas de taxa de sucesso
   - Validar com casos de uso reais (Lenovo, Logitech, BossAuto)

3. **Ajuste fino (se necessário):**
   - Se ainda houver falhas, considerar:
     - Reduzir `SEARCH_ACCEPTANCE_THRESHOLD` de 0.65 para 0.60
     - Reduzir `SEARCH_MIN_MARGIN` de 0.05 para 0.03
     - Ajustar `SEARCH_COMBINED_THRESHOLD` de 0.72 para 0.68

4. **Documentação de casos extremos:**
   - Logos em fundos muito complexos
   - Logos muito pequenos ou muito grandes
   - Logos em ângulos extremos

## Referências

- **Código:** `olinxplus-backend/main.py` (linha 1200-1500)
- **Função:** `adaptive_center_out_crop_pil()`
- **Endpoint:** `POST /search-logo/`
- **Logs:** Digital Ocean App Platform

---

**Última atualização:** 30/11/2025  
**Status:** Aguardando deploy e validação em produção
