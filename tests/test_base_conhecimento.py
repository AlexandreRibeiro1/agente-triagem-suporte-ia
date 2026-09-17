from src.base_conhecimento import LIMIAR_PADRAO, BaseConhecimento
from tests.conftest import CAMINHO_FAQ


def test_pergunta_identica_a_faq_tem_similaridade_maxima(base_conhecimento):
    resultado = base_conhecimento.buscar("Não recebi o e-mail de confirmação do meu cadastro")
    assert resultado["id"] == 3
    assert resultado["similaridade"] == 1.0


def test_encontra_item_por_pergunta_parecida(base_conhecimento):
    resultado = base_conhecimento.buscar("Quero cancelar minha assinatura no fim do mês")
    assert resultado["id"] == 13
    assert resultado["categoria"] == "assinatura"


def test_fora_do_escopo_retorna_none(base_conhecimento):
    assert base_conhecimento.buscar("Meu cachorro comeu meu carregador de notebook") is None


def test_limiar_padrao_barra_resposta_errada(base_conhecimento):
    # Com limiar 0.30 este chamado casava com a FAQ de cartão de pagamento (id 17).
    chamado = "Vocês têm integração com sistema de folha de pagamento?"
    assert base_conhecimento.buscar(chamado) is None

    permissiva = BaseConhecimento(CAMINHO_FAQ, limiar_similaridade=0.30)
    assert permissiva.buscar(chamado)["id"] == 17


def test_limiar_padrao_e_usado_por_default(base_conhecimento):
    assert base_conhecimento.limiar_similaridade == LIMIAR_PADRAO


def test_resposta_automatica_inclui_texto_da_faq(base_conhecimento):
    resultado = base_conhecimento.gerar_resposta_automatica("Qual o prazo para meu pedido chegar?")
    assert resultado["id"] == 4
    assert "5 a 10 dias úteis" in resultado["resposta_automatica"]
    assert "atendente humano" in resultado["resposta_automatica"]


def test_resposta_automatica_none_sem_correspondencia(base_conhecimento):
    assert base_conhecimento.gerar_resposta_automatica("previsão do tempo amanhã") is None
