"""
Testes automatizados com pytest para a base de conhecimento, busca RAG e fallback.
"""

import numpy as np
import pytest

from src.data import carregar_base_conhecimento, CAMPOS_OBRIGATORIOS
from src.rag import (
    BuscadorRAG,
    calcular_similaridade_cosseno,
    MENSAGEM_SEM_INFORMACAO,
)
from src.llm import gerar_resposta


@pytest.fixture(scope="module")
def buscador() -> BuscadorRAG:
    """Cria uma única instância do BuscadorRAG compartilhada entre os testes."""
    return BuscadorRAG(limiar=0.45)


def test_base_conhecimento_possui_cerca_de_30_itens() -> None:
    """Verifica se a base JSON carrega corretamente e possui os campos exigidos."""
    dados = carregar_base_conhecimento()
    assert len(dados) == 30

    for item in dados:
        for campo in CAMPOS_OBRIGATORIOS:
            assert campo in item and len(item[campo].strip()) > 0
        # Todas as respostas devem lembrar de confirmar com a prefeitura ou cooperativa local
        resposta_min = item["resposta"].lower()
        assert "prefeitura" in resposta_min and "cooperativa" in resposta_min


def test_calculo_similaridade_cosseno_numpy() -> None:
    """Testa se vetores iguais têm similaridade 1.0 e ortogonais têm 0.0."""
    vetor_pergunta = np.array([1.0, 0.0], dtype=np.float32)
    matriz_base = np.array(
        [
            [1.0, 0.0],  # Idêntico -> cosseno = 1.0
            [0.0, 1.0],  # Ortogonal (90 graus) -> cosseno = 0.0
            [-1.0, 0.0],  # Oposto (180 graus) -> cosseno = -1.0
        ],
        dtype=np.float32,
    )
    scores = calcular_similaridade_cosseno(vetor_pergunta, matriz_base)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)
    assert np.isclose(scores[2], -1.0)


def test_busca_retorna_top_3_ordenado_com_pontuacao(buscador: BuscadorRAG) -> None:
    """Garante que a busca retorna 3 itens ordenados da maior para a menor pontuação."""
    resultados = buscador.buscar("Onde descartar pilhas e baterias usadas?", top_k=3)
    assert len(resultados) == 3

    for item in resultados:
        assert "pontuacao" in item
        assert -1.0 <= item["pontuacao"] <= 1.0001

    assert resultados[0]["pontuacao"] >= resultados[1]["pontuacao"]
    assert resultados[1]["pontuacao"] >= resultados[2]["pontuacao"]
    assert resultados[0]["id"] == "doc_01"


def test_pergunta_fora_do_tema_fica_abaixo_do_limiar(buscador: BuscadorRAG) -> None:
    """
    Se a maior pontuação ficar abaixo do limiar, o bot não deve inventar resposta.
    """
    buscador_rigoroso = BuscadorRAG(itens=buscador.itens, limiar=0.95)
    resultados = buscador_rigoroso.buscar("Qual a receita de bolo de chocolate?")
    confiavel = buscador_rigoroso.tem_confianca_suficiente(resultados)

    assert confiavel is False
    resposta = gerar_resposta(
        pergunta="Qual a receita de bolo de chocolate?",
        trechos_recuperados=resultados,
        confianca_suficiente=confiavel,
    )
    assert resposta == MENSAGEM_SEM_INFORMACAO


def test_fallback_sem_token_retorna_item_mais_parecido(
    buscador: BuscadorRAG, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem HF_TOKEN configurado, deve usar o fallback devolvendo a resposta do Top-1."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    resultados = buscador.buscar("Como descartar óleo de fritura usado?", top_k=3)
    resposta = gerar_resposta(
        pergunta="Como descartar óleo de fritura usado?",
        trechos_recuperados=resultados,
        confianca_suficiente=True,
    )
    assert resultados[0]["resposta"] in resposta
    assert resultados[0]["categoria"] in resposta
