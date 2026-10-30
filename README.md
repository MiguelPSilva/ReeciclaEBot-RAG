---
title: ReeciclaEBot RAG
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# ♻️ ReeciclaEBot-RAG

O **ReeciclaEBot-RAG** é um chatbot educativo em português sobre **descarte correto e reciclagem de resíduos sólidos no Brasil**, desenvolvido como projeto de portfólio para Engenharia de Inteligência Artificial.

O sistema utiliza a arquitetura **RAG (*Retrieval-Augmented Generation* / Geração Aumentada por Recuperação)** para responder dúvidas do cidadão com base em uma base de conhecimento estruturada (fundamentada na **Política Nacional de Resíduos Sólidos — Lei nº 12.305/2010**, decretos de logística reversa e normas do CONAMA), evitando alucinações e sempre orientando o usuário a confirmar os pontos de coleta com a prefeitura ou cooperativa local.

---

## 🏗️ Arquitetura do Sistema (Diagrama em Texto)

```text
[Pergunta do Usuário]
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 1. RECUPERAÇÃO (src/rag.py)                              │
│  • Gera embedding da pergunta (sentence-transformers)    │
│    Modelo: paraphrase-multilingual-MiniLM-L12-v2         │
│  • Calcula similaridade de cosseno (numpy) contra os     │
│    30 documentos de data/base_conhecimento.json          │
│  • Seleciona os Top-3 trechos com maior pontuação        │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────┐
         │ Maior pontuação >= Limiar (0.45)?│
         └───────┬─────────────────┬───────┘
             NÃO │                 │ SIM
                 ▼                 ▼
┌──────────────────────┐  ┌────────────────────────────────────────────┐
│ Responde que não tem │  │ 2. GERAÇÃO (src/llm.py)                    │
│ essa informação na   │  │  • Se HF_TOKEN válido: envia Top-3 trechos │
│ base (sem inventar!) │  │    ao InferenceClient (Qwen2.5-7B-Instruct)│
└──────────────────────┘  │    exigindo resposta fiel e citação da     │
                          │    categoria.                              │
                          │  • Se sem HF_TOKEN ou erro de API:         │
                          │    ativa FALLBACK e retorna a resposta     │
                          │    oficial do item Top-1 mais parecido.    │
                          └─────────────────────┬──────────────────────┘
                                                │
                                                ▼
                          ┌────────────────────────────────────────────┐
                          │ 3. INTERFACE GRADIO (app.py)               │
                          │  • Exibe a resposta final                  │
                          │  • Mostra os 3 trechos usados e suas       │
                          │    pontuações de similaridade              │
                          └────────────────────────────────────────────┘
```

---

## 📂 Estrutura de Pastas

```text
ReeciclaEBot-RAG/
├── app.py                      # Interface web Gradio (pergunta, resposta e trechos/scores)
├── requirements.txt            # Dependências do projeto
├── .env.example                # Exemplo de configuração de variáveis de ambiente
├── .gitignore                  # Arquivos ignorados pelo Git (como .env e venv/)
├── README.md                   # Documentação e metadados para Hugging Face Spaces
├── data/
│   └── base_conhecimento.json  # 30 itens sobre 12 categorias de resíduos no Brasil
├── src/
│   ├── data.py                 # Leitura, validação e formatação da base JSON
│   ├── rag.py                  # Geração de embeddings e similaridade de cosseno (NumPy)
│   └── llm.py                  # Integração com Hugging Face InferenceClient e fallback
├── eval/
│   ├── perguntas_teste.json    # 20 perguntas escritas de forma diferente das da base
│   └── avaliar.py              # Script que calcula a taxa de acerto Top-1 e Top-3
└── tests/
    └── test_rag.py             # Testes automatizados com pytest para a recuperação
```

---

## 🚀 Como Rodar Localmente

### 1. Criar e ativar um ambiente virtual (recomendado)

No Windows (PowerShell):
```powershell
cd "c:\Python 2026\ReeciclaEBot-RAG"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux / macOS:
```bash
cd ReeciclaEBot-RAG
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Iniciar a interface Gradio

```bash
python app.py
```
O terminal mostrará um endereço local (geralmente `http://127.0.0.1:7860`). Abra-o no navegador para conversar com o bot e inspecionar as pontuações de similaridade.

---

## 🔑 Como Configurar o `HF_TOKEN`

O projeto segue boas práticas de segurança: **nenhum token ou chave secreta fica salvo no código**.

1. Crie uma conta gratuita em [huggingface.co](https://huggingface.co/) e gere um token do tipo **Read** em [Settings -> Access Tokens](https://huggingface.co/settings/tokens).
2. Na raiz do projeto, faça uma cópia do arquivo `.env.example` com o nome `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edite o arquivo `.env` e cole o seu token:
   ```env
   HF_TOKEN=hf_seu_token_real_aqui
   HF_MODEL=Qwen/Qwen2.5-7B-Instruct
   ```
4. **Modo Fallback Automático:** Se você não configurar o `HF_TOKEN` (ou estiver sem internet), o sistema continua funcionando normalmente! Ele detecta a ausência do token e devolve diretamente a resposta oficial do item mais semelhante encontrado pelo RAG.
5. **No Hugging Face Spaces:** Adicione `HF_TOKEN` em *Settings -> Variables and secrets -> New secret*.

---

## 📊 Resultados da Avaliação e Testes

Para avaliar a qualidade do buscador semântico (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + similaridade de cosseno), criamos o arquivo `eval/perguntas_teste.json` com **20 perguntas reformuladas** (usando sinônimos e situações cotidianas diferentes do texto original da base).

Para reproduzir a avaliação e os testes automatizados:

```bash
# Rodar os testes unitários com pytest
pytest tests/ -v

# Rodar o script de métricas de recuperação (Top-1 e Top-3)
python eval/avaliar.py
```

### Números Reais Obtidos (`eval/avaliar.py`)

<!-- RESULTADOS_AVALIACAO_INICIO -->
- **Total de perguntas de teste (`eval/perguntas_teste.json`):** 20
- **Taxa de acerto no Top-1:** **17/20 (85.0%)**
- **Taxa de acerto no Top-3:** **18/20 (90.0%)**
- **Testes automatizados (`pytest tests/ -v`):** **5/5 passando (100%)**

> **Insight prático de Engenharia de IA obtido na avaliação:**
> Na primeira versão de `montar_texto_para_embedding` (`src/data.py`), concatenávamos o texto inteiro da resposta (que inclui avisos legais repetidos em todos os 30 itens, como a Lei 12.305/2010 e a orientação de procurar a prefeitura/cooperativa). Isso diluía o vetor semântico e gerava **75.0% (15/20) no Top-1** e **85.0% (17/20) no Top-3**. Ao ajustar o texto de indexação para focar na `pergunta + categoria + primeira frase da resposta`, o **Top-1 subiu para 85.0% (17/20)** e o **Top-3 subiu para 90.0% (18/20)**.
<!-- RESULTADOS_AVALIACAO_FIM -->

---

## ⚠️ Limitações Atuais

1. **Base de conhecimento estática e enxuta:** Conta atualmente com 30 itens gerais baseados na legislação federal (Lei 12.305/2010 e normas CONAMA).
2. **Regras municipais variáveis:** Horários de caminhão de coleta seletiva, endereços de Ecopontos e cooperativas variam em cada município brasileiro (por isso o bot sempre orienta o usuário a consultar a prefeitura ou cooperativa local e marca com `TODO` os pontos de revisão humana).
3. **Busca apenas densa (vetorial):** Modelos de embedding pequenos podem confundir termos técnicos muito específicos ou siglas raras se não houver busca por palavra-chave combinada.

---

## 🛣️ Próximos Passos (Evolução do Projeto)

- **Busca Híbrida (BM25 + Embeddings):** Combinar busca léxica (palavras exatas) com busca semântica vetorial.
- **Reranking (Cross-Encoder):** Adicionar uma segunda etapa leve para reordenar os Top-3 trechos antes de enviá-los ao LLM.
- **Integração por CEP / Município:** Conectar a uma API ou tabela de dados abertos de prefeituras para listar Ecopontos reais da cidade do usuário.
- **Avaliação de Geração (RAGAS / LLM-as-a-Judge):** Medir também a fidelidade (*faithfulness*) das respostas geradas pelo LLM.

