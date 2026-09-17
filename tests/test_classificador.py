import pytest

from src.classificador import DADOS_TREINO, ClassificadorUrgencia


def test_prever_antes_de_treinar_da_erro():
    with pytest.raises(RuntimeError):
        ClassificadorUrgencia().prever("Sistema fora do ar")


def test_base_de_treino_balanceada():
    rotulos = [rotulo for _, rotulo in DADOS_TREINO]
    assert set(rotulos) == {"urgente", "normal"}
    assert rotulos.count("urgente") == rotulos.count("normal")


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("Servidor caiu e ninguém consegue acessar o sistema", "urgente"),
        ("Falha de segurança crítica, dados expostos", "urgente"),
        ("Como faço para alterar minha senha?", "normal"),
        ("Gostaria de sugerir uma melhoria no relatório", "normal"),
    ],
)
def test_classifica_casos_claros(classificador, texto, esperado):
    assert classificador.prever(texto) == esperado


def test_treinar_com_dados_customizados():
    dados = [
        ("fogo no servidor", "urgente"),
        ("servidor em chamas", "urgente"),
        ("dúvida sobre fatura", "normal"),
        ("pergunta sobre fatura", "normal"),
    ]
    clf = ClassificadorUrgencia()
    clf.treinar(dados, verbose=False)
    assert clf.prever("fogo no servidor agora") == "urgente"
    assert clf.prever("dúvida na fatura") == "normal"
