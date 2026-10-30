"""
Interface web interativa do ReeciclaEBot-RAG construída com Gradio.
Mantém a tela principal limpa para o usuário final e esconde os detalhes técnicos
(trechos e pontuações) em uma aba recolhida (fechada por padrão) para uso apenas em entrevistas.
"""

from typing import List, Dict, Any, Tuple
import gradio as gr

from src.rag import BuscadorRAG, LIMIAR_PADRAO
from src.llm import gerar_resposta

# Inicializa o buscador uma única vez ao subir a aplicação
BUSCADOR: BuscadorRAG | None = None


def obter_buscador() -> BuscadorRAG:
    """Retorna a instância do buscador RAG, inicializando-a na primeira chamada."""
    global BUSCADOR
    if BUSCADOR is None:
        BUSCADOR = BuscadorRAG(limiar=LIMIAR_PADRAO)
    return BUSCADOR


def formatar_trechos_para_exibicao(
    trechos: List[Dict[str, Any]], limiar: float
) -> str:
    """
    Formata os 3 itens mais parecidos com suas pontuações (visível apenas se abrir a aba técnica).
    """
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


def responder_usuario(pergunta: str, limiar: float = LIMIAR_PADRAO) -> Tuple[str, str]:
    """
    Função principal chamada pelo Gradio quando o usuário envia uma pergunta.
    """
    if not pergunta or not pergunta.strip():
        return "Por favor, digite uma dúvida sobre descarte ou reciclagem.", ""

    buscador = obter_buscador()
    buscador.limiar = float(limiar)

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
    """Monta a tela limpa do Gradio focada no usuário final."""
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
            label="Perguntas frequentes (clique para testar):",
        )

        # Aba oculta por padrão: o usuário comum não vê as notas, mas você pode abrir na entrevista!
        with gr.Accordion(
            "⚙️ Detalhes Técnicos do RAG (Oculto para o usuário — abrir apenas para inspeção técnica)",
            open=False,
        ):
            controle_limiar = gr.Slider(
                minimum=0.10,
                maximum=0.85,
                value=LIMIAR_PADRAO,
                step=0.05,
                label="Limiar mínimo de similaridade (Cosine Similarity Threshold)",
            )
            painel_explicacao = gr.Markdown(
                value="*Envie uma pergunta para inspecionar os 3 documentos recuperados e suas pontuações.*"
            )

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
