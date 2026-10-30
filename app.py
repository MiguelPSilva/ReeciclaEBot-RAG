"""
Interface web do ReeciclaEBot-RAG usando Gradio.
"""

from typing import List, Dict, Any, Tuple
import gradio as gr

try:
    import spaces  # Usado apenas quando hospedado em ZeroGPU no Hugging Face Spaces
    decorador_gpu = spaces.GPU
except ImportError:
    def decorador_gpu(func):  # No computador local ou CPU basic, não altera a função
        return func

from src.rag import BuscadorRAG, LIMIAR_PADRAO
from src.llm import gerar_resposta

# Guarda a instância do buscador para não recarregar o modelo a cada pergunta
BUSCADOR: BuscadorRAG | None = None


def obter_buscador() -> BuscadorRAG:
    """Inicializa o BuscadorRAG apenas na primeira vez que for chamado."""
    global BUSCADOR
    if BUSCADOR is None:
        BUSCADOR = BuscadorRAG(limiar=LIMIAR_PADRAO)
    return BUSCADOR


def formatar_trechos_para_exibicao(
    trechos: List[Dict[str, Any]], limiar: float
) -> str:
    """Formata os 3 documentos recuperados e suas pontuações em Markdown."""
    if not trechos:
        return "Nenhum trecho recuperado."

    maior_nota = trechos[0]["pontuacao"]
    status = (
        f"✅ Acima do limiar ({maior_nota:.3f} >= {limiar:.2f})"
        if maior_nota >= limiar
        else f"⚠️ Abaixo do limiar ({maior_nota:.3f} < {limiar:.2f})"
    )

    linhas = [f"**Status da Recuperação:** {status}\n"]
    for pos, item in enumerate(trechos, start=1):
        bloco = (
            f"**#{pos} — ID: `{item['id']}` | Categoria: {item['categoria']} "
            f"| Similaridade: `{item['pontuacao']:.3f}`**\n"
            f"- **Pergunta na base:** {item['pergunta']}\n"
            f"- **Resposta na base:** {item['resposta']}"
        )
        linhas.append(bloco)

    return "\n\n---\n\n".join(linhas)


@decorador_gpu
def responder_usuario(pergunta: str, limiar: float = LIMIAR_PADRAO) -> Tuple[str, str]:
    """Executa a busca RAG e gera a resposta para a interface."""
    if not pergunta or not pergunta.strip():
        return "Por favor, digite uma dúvida sobre descarte ou reciclagem.", ""

    buscador = obter_buscador()
    buscador.limiar = float(limiar)

    # Busca os 3 itens mais parecidos e verifica se passou da nota mínima
    trechos = buscador.buscar(pergunta, top_k=3)
    confiavel = buscador.tem_confianca_suficiente(trechos)

    resposta = gerar_resposta(
        pergunta=pergunta,
        trechos_recuperados=trechos,
        confianca_suficiente=confiavel,
    )
    painel_trechos = formatar_trechos_para_exibicao(trechos, buscador.limiar)
    return resposta, painel_trechos


def construir_interface() -> gr.Blocks:
    """Cria os componentes visuais da página no Gradio."""
    with gr.Blocks(title="ReeciclaEBot-RAG ♻️") as demo:
        gr.Markdown(
            "# ♻️ ReeciclaEBot\n"
            "Tire suas dúvidas sobre **descarte correto e reciclagem de resíduos no Brasil** "
            "(com base na Política Nacional de Resíduos Sólidos - Lei 12.305/2010)."
        )

        campo_pergunta = gr.Textbox(
            label="Qual é a sua dúvida sobre reciclagem ou descarte?",
            placeholder="Ex.: O que eu faço com caixas de pizza que acabei de comer?",
            lines=2,
        )

        botao_enviar = gr.Button("Perguntar ao ReeciclaEBot", variant="primary")

        campo_resposta = gr.Textbox(
            label="Orientação de Descarte",
            lines=6,
            interactive=False,
        )

        gr.Examples(
            examples=[
                ["O que eu faço com lixo de caixas de pizza que acabei de comer?"],
                ["O que fazer com óleo de fritura usado?"],
                ["Como descartar remédios vencidos com segurança?"],
                ["Como embalar copo de vidro quebrado para não machucar o gari?"],
                ["Onde descartar pilhas alcalinas e bateria de celular inchada?"],
            ],
            inputs=[campo_pergunta],
            label="Perguntas frequentes:",
        )

        # Seção recolhida por padrão para inspecionar os trechos e scores do RAG
        with gr.Accordion(
            "Detalhes da busca (trechos recuperados e pontuações)",
            open=False,
        ):
            controle_limiar = gr.Slider(
                minimum=0.10,
                maximum=0.85,
                value=LIMIAR_PADRAO,
                step=0.05,
                label="Limiar mínimo de similaridade",
            )
            painel_explicacao = gr.Markdown(
                value="*Envie uma pergunta para ver os 3 trechos mais próximos e suas pontuações.*"
            )

        # Conecta o clique do botão e o Enter do teclado à função responder_usuario
        botao_enviar.click(
            fn=responder_usuario,
            inputs=[campo_pergunta, controle_limiar],
            outputs=[campo_resposta, painel_explicacao],
        )
        campo_pergunta.submit(
            fn=responder_usuario,
            inputs=[campo_pergunta, controle_limiar],
            outputs=[campo_resposta, painel_explicacao],
        )

    return demo


if __name__ == "__main__":
    app = construir_interface()
    app.launch()
