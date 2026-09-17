import pandas as pd

from src.avaliacao import (
    avaliar_classificador,
    avaliar_retriever,
    carregar_casos_rag,
    varrer_limiares,
)
from tests.conftest import CAMINHO_CASOS_RAG


def test_classificador_supera_baseline():
    resultado = avaliar_classificador(n_repeats=3)

    assert resultado["n_dobras"] == 12
    assert resultado["acuracia"][0] > resultado["baseline_acuracia"][0]
    assert resultado["f1_macro"][0] > resultado["baseline_f1_macro"][0]


def test_matriz_de_confusao_soma_todas_as_previsoes():
    resultado = avaliar_classificador(n_repeats=2)

    # Cada exemplo aparece uma vez no teste por repetição
    assert resultado["matriz_confusao"].to_numpy().sum() == resultado["n_exemplos"] * 2
    assert list(resultado["por_classe"].index) == ["urgente", "normal"]


def test_casos_rag_tem_itens_dentro_e_fora_do_escopo():
    casos = carregar_casos_rag(CAMINHO_CASOS_RAG)

    assert casos["id_esperado"].notna().sum() > 0
    assert casos["id_esperado"].isna().sum() > 0


def test_avaliar_retriever_conta_acertos_recusas_e_erros(base_conhecimento):
    casos = pd.DataFrame({
        "chamado": [
            "Qual o prazo para meu pedido chegar?",      # acerta o item 4
            "Qual o prazo para meu pedido chegar?",      # anotado errado de propósito
            "Meu cachorro comeu o carregador",           # recusa correta
        ],
        "id_esperado": pd.array([4, 1, None], dtype="Int64"),
    })

    resultado = avaliar_retriever(base_conhecimento, casos, limiar=0.5)

    assert resultado["acerto_no_escopo"] == 0.5
    assert resultado["recusa_fora_escopo"] == 1.0
    assert resultado["respostas_automaticas"] == 2
    assert resultado["resposta_errada"] == 0.5


def test_avaliacao_nao_altera_limiar_da_base(base_conhecimento):
    limiar_original = base_conhecimento.limiar_similaridade
    casos = carregar_casos_rag(CAMINHO_CASOS_RAG)

    varrer_limiares(base_conhecimento, casos, [0.1, 0.9])

    assert base_conhecimento.limiar_similaridade == limiar_original


def test_limiar_mais_alto_nunca_gera_mais_respostas(base_conhecimento):
    casos = carregar_casos_rag(CAMINHO_CASOS_RAG)

    tabela = varrer_limiares(base_conhecimento, casos, [0.1, 0.3, 0.5, 0.7])

    assert tabela["respostas_automaticas"].is_monotonic_decreasing
