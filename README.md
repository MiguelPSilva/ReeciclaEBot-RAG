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

O sistema utiliza a arquitetura **RAG (*Retrieval-Augmented Generation* / Geração Aumentada por Recuperação)** para responder dúvidas com base em uma base de conhecimento estruturada (fundamentada na **Política Nacional de Resíduos Sólidos — Lei nº 12.305/2010**, decretos de logística reversa e normas do CONAMA), evitando alucinações e sempre orientando o usuário a confirmar os pontos de coleta com a prefeitura ou cooperativa local.

---

## 🏗️ Arquitetura do Sistema

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
│ essa informação na   │  │  • Se HF_TOKEN ativo: envia Top-3 trechos  │
│ base (sem inventar)  │  │    ao InferenceClient (Llama-3.1-8B)       │
└──────────────────────┘  │    pedindo resposta baseada nos trechos    │
                          │    e citação da categoria.                 │
                          │  • Se sem HF_TOKEN ou falha na API:        │
                          │    ativa FALLBACK e retorna a resposta     │
                          │    direta do item Top-1 mais parecido.     │
                          └─────────────────────┬──────────────────────┘
                                                │
                                                ▼
                          ┌────────────────────────────────────────────┐
                          │ 3. INTERFACE GRADIO (app.py)               │
                          │  • Exibe a orientação de descarte          │
                          │  • Aba recolhida com os trechos usados e   │
                          │    as pontuações de similaridade           │
                          └────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto

```text
ReeciclaEBot-RAG/
├── app.py                      # Interface web Gradio
├── requirements.txt            # Dependências do projeto
├── .env.example                # Exemplo de configuração de variáveis de ambiente
├── .gitignore                  # Arquivos ignorados pelo Git
├── README.md                   # Documentação do projeto
├── data/
│   └── base_conhecimento.json  # 30 itens sobre 12 categorias de resíduos no Brasil
├── src/
│   ├── data.py                 # Leitura, validação e preparação da base JSON
│   ├── rag.py                  # Geração de embeddings e similaridade de cosseno (NumPy)
│   └── llm.py                  # Integração com Hugging Face InferenceClient e fallback
├── eval/
│   ├── perguntas_teste.json    # 20 perguntas reformuladas para avaliação
│   └── avaliar.py              # Cálculo da taxa de acerto Top-1 e Top-3
└── tests/
    └── test_rag.py             # Testes automatizados com pytest
```

---

## 🚀 Como Rodar Localmente

### 1. Criar e ativar um ambiente virtual

No Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux / macOS:
```bash
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
Acesse o endereço local exibido no terminal (`http://127.0.0.1:7860`).

---

## 🔑 Como Configurar o `HF_TOKEN`

Nenhum token fica salvo no código. Para usar a geração via API da Hugging Face:

1. Crie uma cópia do arquivo `.env.example` com o nome `.env`:
   ```bash
   cp .env.example .env
   ```
2. Preencha o seu token no arquivo `.env`:
   ```env
   HF_TOKEN=hf_seu_token_aqui
   HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
   ```
3. **Modo Fallback:** Se `HF_TOKEN` não estiver configurado ou a chamada à API falhar, o sistema retorna automaticamente a resposta oficial do item mais semelhante encontrado na base (`Top-1`).
4. **No Hugging Face Spaces:** Configure `HF_TOKEN` em *Settings -> Variables and secrets -> New secret*.

---

## 📊 Resultados da Avaliação e Testes

A avaliação da recuperação foi feita com o arquivo `eval/perguntas_teste.json`, contendo **20 perguntas escritas de forma diferente das perguntas da base**.

Comandos para rodar os testes e a avaliação:

```bash
pytest tests/ -v
python eval/avaliar.py
```

### Métricas Obtidas (`eval/avaliar.py`)

- **Total de perguntas de teste:** 20
- **Taxa de acerto no Top-1:** **17/20 (85.0%)**
- **Taxa de acerto no Top-3:** **18/20 (90.0%)**
- **Testes automatizados (`pytest tests/ -v`):** **5/5 passando (100%)**

---

## ⚠️ Limitações

1. **Base de conhecimento enxuta:** Conta com 30 itens gerais baseados na legislação federal (Lei 12.305/2010 e normas CONAMA).
2. **Regras municipais variáveis:** Dias de coleta seletiva, endereços de Ecopontos e materiais aceitos variam conforme a cidade ou cooperativa local (por isso todas as respostas orientam a confirmação local e possuem marcações `TODO` para revisão).
3. **Busca apenas semântica:** Erros graves de digitação ou siglas muito específicas podem reduzir a pontuação de similaridade se não houver busca léxica combinada.

---

## 🛣️ Próximos Passos

- **Busca Híbrida (BM25 + Embeddings):** Combinar busca por palavras-chave exatas com busca vetorial.
- **Reranking (Cross-Encoder):** Reordenar os 3 trechos recuperados com um modelo Cross-Encoder antes da etapa de geração.
- **Consulta por Município/CEP:** Integrar dados abertos municipais para indicar Ecopontos próximos ao usuário.
