# L7 - Exemplos de Configuração para Múltiplos Providers

A camada L7 agora suporta múltiplos providers para geração automática de texto final. Aqui estão exemplos de como configurar cada um.

## 1. Provider: Ollama (Modelos Locais)

**Vantagens:**
- Modelos rodando localmente (sem conexão à internet)
- Controle total sobre o modelo
- Custos zero depois de instalado

**Configuração em `config.yaml`:**
```yaml
l7:
  provider: "ollama"
  model: "llama2"              # ou mistral, neural-chat, etc.
  temperature: 0.7             # Criatividade (0.0-1.0)
  max_tokens: 4096             # Limite de tokens
```

**Como usar:**

1. **Instale Ollama:**
   ```bash
   # De: https://ollama.ai
   # Ou via brew: brew install ollama
   ```

2. **Puxe um modelo:**
   ```bash
   ollama pull llama2           # ~4GB
   ollama pull mistral          # ~5GB (melhor qualidade)
   ollama pull neural-chat      # ~4GB
   ```

3. **Inicie o servidor (rodando automaticamente):**
   ```bash
   ollama serve                 # Listening on 127.0.0.1:11434
   ```

4. **Execute o pipeline:**
   ```python
   from pipeline import HybridLLMPipeline
   from config_loader import load_config
   
   config = load_config()
   pipeline = HybridLLMPipeline(config=config)
   result = pipeline.process("Sua pergunta aqui")
   print(result.response)
   ```

---

## 2. Provider: Groq (Cloud API)

**Vantagens:**
- Modelos poderosos (Mixtral, Llama, etc.)
- Muito rápido (inferência acelerada)
- API gratuita com limite de taxa

**Configuração em `config.yaml`:**
```yaml
generation:
  provider: "groq"
  groq_model: "mixtral-8x7b-32768"  # ou llama2-70b, etc.

l7:
  provider: "groq"
  groq_model: "mixtral-8x7b-32768"
```

**Como usar:**

1. **Obtenha uma API Key:**
   - Visite: https://console.groq.com/
   - Crie uma conta e obtenha a chave

2. **Configure a variável de ambiente:**
   ```bash
   # Windows (PowerShell)
   $env:GROQ_API_KEY = "sua-chave-aqui"
   
   # Linux/Mac
   export GROQ_API_KEY="sua-chave-aqui"
   ```

3. **Execute o pipeline:**
   ```python
   config = load_config()
   pipeline = HybridLLMPipeline(config=config)
   result = pipeline.process("Sua pergunta aqui")
   ```

**Modelos disponíveis:**
- `mixtral-8x7b-32768` - Melhor balanceamento qualidade/velocidade
- `llama2-70b-4096` - Mais poderoso, contexto menor
- `gemma-7b-it` - Compacto e rápido

---

## 3. Provider: Custom LM (Modelo Customizado)

**Vantagens:**
- Usar modelos finetuned próprios
- Máximo controle e customização
- Offline após treinamento

**Configuração em `config.yaml`:**
```yaml
l7:
  provider: "custom_lm"
  custom_lm_path: "./models/meu-modelo-l7-fine-tuned"
```

**Como usar:**

1. **Treinar ou preparar o modelo:**
   ```python
   # Seu código de treinamento...
   model.save_pretrained("./models/meu-modelo-l7-fine-tuned")
   ```

2. **Configure no YAML:**
   ```yaml
   l7:
     provider: "custom_lm"
     custom_lm_path: "./models/meu-modelo-l7-fine-tuned"
   ```

3. **Execute:**
   ```python
   config = load_config()
   pipeline = HybridLLMPipeline(config=config)
   result = pipeline.process("Sua pergunta aqui")
   ```

---

## 4. Provider: Template (Fallback - sem LLM)

**Vantagens:**
- Não requer nenhuma dependência
- Rápido e confiável
- Bom para testes

**Configuração em `config.yaml`:**
```yaml
l7:
  provider: "template"
```

**Comportamento:**
- Retorna o melhor texto disponível das camadas anteriores (L6, L5, L4)
- Sem geração de novo conteúdo via LLM

---

## Comparação de Providers

| Provider | Velocidade | Qualidade | Custo | Online | Instalação |
|----------|-----------|-----------|-------|--------|------------|
| Ollama | Rápido | Boa | Grátis | Não | Fácil |
| Groq | Muito Rápido | Excelente | Grátis* | Sim | Nenhuma |
| Custom LM | Variável | Customizável | Grátis | Não | Complexa |
| Template | Instant | Boa | Grátis | N/A | Nenhuma |

*Groq tem limite de taxa (30 requisições/minuto para usuários gratuitos)

---

## Exemplo Completo: Usar Ollama com Mistral

**Arquivo `config.yaml`:**
```yaml
l7:
  provider: "ollama"
  model: "mistral"
  temperature: 0.7
  max_tokens: 2048

# Configuração geral do pipeline
generation:
  provider: "template"  # L5 sem geração
  
finalization:
  provider: "template"  # L6 sem geração
```

**Código Python:**
```python
#!/usr/bin/env python3
from pipeline import HybridLLMPipeline
from config_loader import load_config

# Carrega configuração
config = load_config()

# Inicializa pipeline com Ollama/Mistral
pipeline = HybridLLMPipeline(config=config, verbose=True)

# Processa pergunta
pergunta = "Qual é a relação entre lógica paraconsistente e epistemologia?"
resultado = pipeline.process(pergunta)

print("\n" + "="*70)
print("RESPOSTA FINAL (L7):")
print("="*70)
print(resultado.response)
print("\nValor de Verdade:", resultado.truth_value)
print("Certeza:", resultado.certainty)
```

**Fluxo:**
1. L1: Extrai conceitos da pergunta
2. L2: Aplica juízos kantianos
3. L3: Análise paraconsistente
4. L4: Síntese russelliana
5. L6: Refinamento
6. **L7 (Ollama/Mistral): Gera texto final automático e fluido**

---

## Dicas e Troubleshooting

### Ollama não conecta
```python
# Verifique se está rodando
import ollama
try:
    ollama.list()
    print("✓ Ollama conectado")
except:
    print("✗ Ollama não está rodando")
    print("Inicie com: ollama serve")
```

### Groq retorna erro de limite
- Aguarde alguns segundos
- Use um modelo mais rápido (gemma-7b-it)
- Configure cache/batch processing

### Custom LM não carrega
- Verifique o caminho: `os.path.exists(custom_lm_path)`
- Certifique-se que é um modelo Hugging Face válido

### Textos muito curtos/longos
Ajuste `max_tokens` em `config.yaml`:
```yaml
l7:
  max_tokens: 8192  # Aumentar para textos mais longos
```

---

## Integração Programática

Passar parâmetros diretamente ao `process()`:

```python
result = pipeline.process(
    prompt="Sua pergunta",
    # L7 específico
)

# Ou ao finalize_text:
l7_engine = FinalTextEngine(config=config)
texto_final = l7_engine.finalize_text(
    prompt="Sua pergunta",
    l1_summary="Conceitos...",
    l2_summary="Juízos...",
    l3_summary="Análise...",
    l4_response="Síntese...",
    l5_text="Geração...",
    l6_text="Refinamento...",
    provider="ollama",
    model="mistral",
    temperature=0.8,
    max_tokens=4096
)
```

---

## Recomendações

1. **Para desenvolvimento local:** Use **Ollama + Mistral**
   - Rápido, offline, boa qualidade

2. **Para produção:** Use **Groq API**
   - Rápido, confiável, escalável

3. **Para testes:** Use **Template**
   - Sem dependências, instant feedback

4. **Para máxima qualidade:** Fine-tune um **Custom LM**
   - Controle total, especifico do domínio
