# =============================================================================
# Agente de Triagem e Atendimento Automático de Suporte com IA (v2 — RAG)
#
# Evolução do desafio final da Trilha Especialista em Inteligência Artificial
# (Alura / Santander Academy). Agora o agente não só classifica a urgência do
# chamado, como também tenta RESOLVER automaticamente chamados normais,
# consultando uma base de conhecimento via RAG (Retrieval-Augmented
# Generation) local, sem custo de API.
#
# Autor: Alexandre (Ale)
# =============================================================================

import os
import pandas as pd

from src.classificador import ClassificadorUrgencia
from src.base_conhecimento import BaseConhecimento
from src.agente import AgenteTriagemSuporte

CAMINHO_BASE_CONHECIMENTO = os.path.join("data", "base_conhecimento.csv")
CAMINHO_SAIDA = os.path.join("data", "relatorio_triagem.csv")


# -----------------------------------------------------------------------------
# 1. Treina o classificador de urgência
# -----------------------------------------------------------------------------
classificador = ClassificadorUrgencia()
classificador.treinar()

# -----------------------------------------------------------------------------
# 2. Carrega a base de conhecimento (FAQ) para o RAG
# -----------------------------------------------------------------------------
base_conhecimento = BaseConhecimento(CAMINHO_BASE_CONHECIMENTO, limiar_similaridade=0.30)

# -----------------------------------------------------------------------------
# 3. Instancia o agente, unindo classificador + RAG
# -----------------------------------------------------------------------------
agente = AgenteTriagemSuporte(classificador, base_conhecimento)

# -----------------------------------------------------------------------------
# 4. Simula a chegada de novos chamados (nunca vistos pelo modelo)
#    Propositalmente incluindo: urgentes, normais com boa correspondência na
#    FAQ, e normais SEM correspondência (para testar o fallback humano).
# -----------------------------------------------------------------------------
novos_chamados = [
    "O site caiu para todos os clientes, ninguém consegue comprar",
    "Como faço para trocar minha senha de acesso?",
    "Identificamos uma falha de segurança grave no login",
    "Qual o prazo para meu pedido chegar?",
    "Gostaria de entender como funciona o plano premium de vocês",
    "Não recebi o e-mail de confirmação do meu cadastro",
    "Toda a equipe está sem acesso ao sistema agora",
    "Vocês têm alguma integração com sistema de folha de pagamento?",  # sem match na FAQ
    "Quero cancelar minha assinatura no fim do mês",
    "Meu cachorro comeu meu carregador de notebook, o que eu faço?",  # fora de escopo, sem match
]

print("\n" + "=" * 70)
print("AGENTE PROCESSANDO NOVOS CHAMADOS")
print("=" * 70)
for chamado in novos_chamados:
    registro = agente.agir(chamado)
    print(f"\n📨 Chamado: {registro['chamado']}")
    print(f"   → Urgência prevista: {registro['urgencia_prevista'].upper()}")
    print(f"   → Ação do agente: {registro['acao_tomada']}")
    if registro["origem_resposta"] == "rag":
        print(f"   → Similaridade com a FAQ: {registro['similaridade_faq']}")
        print(f"   → Resposta automática:\n{registro['resposta_automatica']}")

# -----------------------------------------------------------------------------
# 5. Relatório final automatizado (saída do processo)
# -----------------------------------------------------------------------------
df_log, estatisticas = agente.gerar_relatorio()

print("\n" + "=" * 70)
print("RELATÓRIO FINAL - RESUMO DE TRIAGEM E ATENDIMENTO")
print("=" * 70)
print("\nDistribuição por urgência:")
print(estatisticas["resumo_urgencia"].to_string())
print("\nDistribuição por origem da resposta:")
print(estatisticas["resumo_origem"].to_string())
print(f"\nTotal de chamados processados: {estatisticas['total_chamados']}")
print(f"Taxa de resolução automática (RAG): {estatisticas['taxa_resolucao_automatica']:.0%}")

os.makedirs(os.path.dirname(CAMINHO_SAIDA), exist_ok=True)
df_log.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8-sig")
print(f"\n✅ Relatório salvo em: {CAMINHO_SAIDA}")
