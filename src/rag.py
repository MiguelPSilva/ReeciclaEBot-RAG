"""
Módulo de recuperação (Retrieval) do RAG.
Converte textos em vetores com sentence-transformers e calcula similaridade de cosseno com numpy.
"""

from typing import Any, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data import carregar_base_conhecimento, montar_texto_para_embedding

# Modelo multilíngue leve que gera vetores de 384 dimensões
MODELO_EMBEDDING_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# TODO: Ajustar o limiar caso o modelo de embeddings seja trocado no futuro
LIMIAR_PADRAO = 0.45

MENSAGEM_SEM_INFORMACAO = (
    "Desculpe, não encontrei informações confiáveis na minha base sobre esse assunto. "
    "Por favor, consulte a prefeitura ou a cooperativa de reciclagem da sua cidade."
)


def calcular_similaridade_cosseno(
    vetor_pergunta: np.ndarray, matriz_base: np.ndarray
) -> np.ndarray:
    """
    Calcula a similaridade de cosseno entre a pergunta e todos os itens da base.
    Fórmula matemática: (A . B) / (||A|| * ||B||)
    """
    # Produto escalar entre cada linha da matriz da base e o vetor da pergunta
    produto_escalar = np.dot(matriz_base, vetor_pergunta)

    # Calcula o tamanho (norma) do vetor da pergunta e de cada vetor da base
    norma_pergunta = np.linalg.norm(vetor_pergunta)
    normas_base = np.linalg.norm(matriz_base, axis=1)

    # Usa 1e-10 como valor mínimo no denominador para evitar divisão por zero
    denominador = np.maximum(normas_base * norma_pergunta, 1e-10)
    return produto_escalar / denominador


class BuscadorRAG:
    """Gera os embeddings da base de conhecimento e busca os documentos mais próximos."""

    def __init__(
        self,
        itens: List[Dict[str, str]] | None = None,
        nome_modelo: str = MODELO_EMBEDDING_PADRAO,
        limiar: float = LIMIAR_PADRAO,
    ) -> None:
        self.itens = itens if itens is not None else carregar_base_conhecimento()
        self.limiar = limiar
        self.modelo = SentenceTransformer(nome_modelo)
        self.embeddings_base = self._gerar_embeddings_base()

    def _gerar_embeddings_base(self) -> np.ndarray:
        """Transforma todos os documentos da base em uma matriz de vetores numpy."""
        textos = [montar_texto_para_embedding(item) for item in self.itens]
        vetores = self.modelo.encode(textos, convert_to_numpy=True)
        return np.asarray(vetores, dtype=np.float32)

    def buscar(self, pergunta: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Busca os `top_k` itens mais parecidos com a pergunta do usuário
        e adiciona a 'pontuacao' de similaridade em cada um.
        """
        if not pergunta or not pergunta.strip():
            return []

        # Converte a pergunta do usuário em vetor numérico
        vetor_pergunta = self.modelo.encode(pergunta.strip(), convert_to_numpy=True)
        vetor_pergunta = np.asarray(vetor_pergunta, dtype=np.float32)

        # Calcula a nota de similaridade com todos os itens da base
        pontuacoes = calcular_similaridade_cosseno(vetor_pergunta, self.embeddings_base)

        # Ordena os índices do maior para o menor score e pega os top_k primeiros
        indices_ordenados = np.argsort(pontuacoes)[::-1][:top_k]

        resultados: List[Dict[str, Any]] = []
        for idx in indices_ordenados:
            item_copia = dict(self.itens[int(idx)])
            item_copia["pontuacao"] = float(pontuacoes[int(idx)])
            resultados.append(item_copia)

        return resultados

    def tem_confianca_suficiente(self, resultados: List[Dict[str, Any]]) -> bool:
        """Retorna True se o item mais parecido atingiu a pontuação mínima (limiar)."""
        if not resultados:
            return False
        return resultados[0]["pontuacao"] >= self.limiar
