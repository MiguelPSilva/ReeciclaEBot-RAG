---
title: ReeciclaEBot RAG
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
license: mit
---

# ♻️ ReeciclaEBot-RAG

🔗 **Demonstração online (Hugging Face Spaces):** https://huggingface.co/spaces/Miguelprsilva/ReeciclaEBot-RAG

O **ReeciclaEBot-RAG** é um chatbot educativo em português sobre **descarte correto e reciclagem de resíduos sólidos no Brasil**, desenvolvido como projeto de portfólio em Engenharia de Inteligência Artificial.

O sistema usa a arquitetura **RAG** (*Retrieval-Augmented Generation*, ou Geração Aumentada por Recuperação) para responder dúvidas com base em uma base de conhecimento estruturada, fundamentada na Política Nacional de Resíduos Sólidos (Lei nº 12.305/2010), em decretos de logística reversa e em normas do CONAMA. A resposta só sai de trechos recuperados da base, o que reduz o risco de alucinação, e o bot sempre orienta o usuário a confirmar os pontos de coleta com a prefeitura ou a cooperativa local.

## ℹ️ Estado atual da demonstração

- **Recuperação semântica:** funcionando e avaliada (resultados abaixo).
- **Geração com LLM:** a camada com Llama 3.1 8B via API do Hugging Face está implementada em `src/llm.py`, mas **não foi validada de ponta a ponta na demonstração pública**, porque a chamada gratuita da API não funcionou com o meu token.
- **Por isso, a demo pública roda em modo fallback:** devolve a resposta do item mais parecido da base (Top-1), sem reescrita por LLM.

## 🏗️ Arquitetura do sistema

```text
[Pergunta do usuário]
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 1. RECUPERAÇÃO (src/rag.py)                              │
│  • Gera o embedding da pergunta (sentence-transformers)  │
│    Modelo: paraphrase-multilingual-MiniLM-L12-v2         │
│  • Calcula a similaridade de cosseno (NumPy) contra os   │
│    30 itens de data/base_conhecimento.json               │
│  • Seleciona os Top-3 trechos com maior pontuação        │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────┐
         │ Maior pontuação >= limiar (0,45)? │
         └───────┬─────────────────┬────────┘
             NÃO │                 │ SIM
                 ▼                 ▼
┌──────────────────────┐  ┌────────────────────────────────────────────┐
│ Responde que não tem │  │ 2. GERAÇÃO (src/llm.py)                    │
│ essa informação na   │  │  • Com HF_TOKEN ativo: envia os Top-3      │
│ base (sem inventar)  │  │    trechos ao InferenceClient (Llama 3.1   │
└──────────────────────┘  │    8B), pedindo resposta baseada neles e   │
                          │    citação da categoria.                   │
                          │  • Sem HF_TOKEN ou com falha na API:       │
                          │    ativa o FALLBACK e retorna a resposta   │
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

## 📂 Estrutura do projeto

```text
ReeciclaEBot-RAG/
├── app.py                      # Interface web Gradio
├── requirements.txt            # Dependências do projeto
├── .env.example                # Exemplo de variáveis de ambiente
├── .gitignore                  # Arquivos ignorados pelo Git
├── README.md                   # Documentação do projeto
├── data/
│   └── base_conhecimento.json  # 30 itens sobre 12 categorias de resíduos no Brasil
├── src/
│   ├── data.py                 # Leitura, validação e preparação da base JSON
│   ├── rag.py                  # Embeddings e similaridade de cosseno (NumPy)
│   └── llm.py                  # Integração com Hugging Face InferenceClient e fallback
├── eval/
│   ├── perguntas_teste.json    # 20 perguntas reformuladas para avaliação
│   └── avaliar.py              # Cálculo da taxa de acerto Top-1 e Top-3
└── tests/
    └── test_rag.py             # Testes automatizados com pytest
```

## 🚀 Como rodar localmente

**1. Criar e ativar um ambiente virtual**

No Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Instalar as dependências**

```bash
pip install -r requirements.txt
```

**3. Iniciar a interface Gradio**

```bash
python app.py
```

Acesse o endereço local exibido no terminal (normalmente http://127.0.0.1:7860).

## 🔑 Como configurar o HF_TOKEN

Nenhum token fica salvo no código. Para usar a geração via API do Hugging Face:

1. Copie o arquivo `.env.example` para `.env`:

   ```bash
   cp .env.example .env
   ```

2. Preencha o token no arquivo `.env`:

   ```env
   HF_TOKEN=hf_seu_token_aqui
   HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
   ```

**Modo fallback:** se o `HF_TOKEN` não estiver configurado ou se a chamada à API falhar, o sistema retorna automaticamente a resposta do item mais semelhante encontrado na base (Top-1).

**No Hugging Face Spaces:** configure o `HF_TOKEN` em *Settings → Variables and secrets → New secret*.

## 📊 Resultados da avaliação e testes

A avaliação da recuperação usa o arquivo `eval/perguntas_teste.json`, com 20 perguntas escritas de forma diferente das perguntas da base.

Para rodar os testes e a avaliação:

```bash
pytest tests/ -v
python eval/avaliar.py
```

**Métricas obtidas** (`eval/avaliar.py`):

| Métrica | Resultado |
|---|---|
| Perguntas de teste | 20 |
| Acerto no Top-1 | 17/20 (85,0%) |
| Acerto no Top-3 | 18/20 (90,0%) |
| Testes automatizados (pytest) | 5/5 passando |

Essas métricas medem apenas a **recuperação** (se o item certo aparece entre os mais parecidos). A qualidade da resposta gerada pelo LLM não foi avaliada.

## ⚠️ Limitações

- **Base de conhecimento enxuta:** 30 itens gerais, baseados na legislação federal (Lei 12.305/2010 e normas do CONAMA).
- **Avaliação pequena:** 20 perguntas dão uma estimativa limitada, e os números podem mudar com um conjunto maior.
- **Regras municipais variam:** dias de coleta seletiva, endereços de Ecopontos e materiais aceitos mudam conforme a cidade ou a cooperativa, por isso todas as respostas orientam a confirmação local.
- **Busca apenas semântica:** erros graves de digitação ou siglas muito específicas podem reduzir a pontuação de similaridade, pois não há busca léxica combinada.
- **Geração com LLM não validada na demo pública** (veja "Estado atual da demonstração").

## 🛣️ Próximos passos

- **Validar a geração com LLM** de ponta a ponta e medir a qualidade das respostas geradas.
- **Busca híbrida (BM25 + embeddings):** combinar busca por palavras-chave exatas com busca vetorial.
- **Reclassificação (Cross-Encoder):** reordenar os 3 trechos recuperados antes da etapa de geração.
- **Consulta por município/CEP:** integrar dados abertos municipais para indicar Ecopontos próximos.

## 📄 Licença

MIT.
