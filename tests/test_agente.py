from src.agente import AgenteTriagemSuporte
from tests.conftest import BaseFixa, ClassificadorFixo

RESULTADO_RAG = {"resposta_automatica": "Resposta da FAQ", "similaridade": 0.9}


def test_urgente_escala_sem_consultar_faq():
    base = BaseFixa(RESULTADO_RAG)
    agente = AgenteTriagemSuporte(ClassificadorFixo("urgente"), base)

    registro = agente.agir("Sistema fora do ar")

    assert registro["origem_resposta"] == "escalonamento"
    assert registro["resposta_automatica"] is None
    assert base.consultas == 0


def test_normal_com_match_responde_via_rag():
    agente = AgenteTriagemSuporte(ClassificadorFixo("normal"), BaseFixa(RESULTADO_RAG))

    registro = agente.agir("Como troco a senha?")

    assert registro["origem_resposta"] == "rag"
    assert registro["resposta_automatica"] == "Resposta da FAQ"
    assert registro["similaridade_faq"] == 0.9


def test_normal_sem_match_vai_para_fila_humana():
    agente = AgenteTriagemSuporte(ClassificadorFixo("normal"), BaseFixa(None))

    registro = agente.agir("Pergunta sem resposta na FAQ")

    assert registro["origem_resposta"] == "fila_humana"
    assert registro["resposta_automatica"] is None


def test_agir_classifica_uma_unica_vez():
    classificador = ClassificadorFixo("normal")
    agente = AgenteTriagemSuporte(classificador, BaseFixa(RESULTADO_RAG))

    agente.agir("Como troco a senha?")

    assert classificador.chamadas == 1


def test_relatorio_calcula_taxa_de_resolucao():
    agente = AgenteTriagemSuporte(ClassificadorFixo("normal"), BaseFixa(RESULTADO_RAG))
    agente.agir("a")
    agente.agir("b")
    agente.base_conhecimento = BaseFixa(None)
    agente.agir("c")
    agente.classificador = ClassificadorFixo("urgente")
    agente.agir("d")

    df, estatisticas = agente.gerar_relatorio()

    assert len(df) == 4
    assert estatisticas["total_chamados"] == 4
    assert estatisticas["taxa_resolucao_automatica"] == 0.5
    assert estatisticas["resumo_origem"].to_dict() == {"rag": 2, "fila_humana": 1, "escalonamento": 1}


def test_relatorio_vazio_nao_quebra():
    agente = AgenteTriagemSuporte(ClassificadorFixo("normal"), BaseFixa(None))

    df, estatisticas = agente.gerar_relatorio()

    assert df.empty
    assert estatisticas["taxa_resolucao_automatica"] == 0


def test_pipeline_real_ponta_a_ponta(classificador, base_conhecimento):
    agente = AgenteTriagemSuporte(classificador, base_conhecimento)

    assert agente.agir("O site caiu para todos os clientes")["origem_resposta"] == "escalonamento"
    assert agente.agir("Qual o prazo para meu pedido chegar?")["origem_resposta"] == "rag"
    assert agente.agir("Meu cachorro comeu o carregador")["origem_resposta"] == "fila_humana"
