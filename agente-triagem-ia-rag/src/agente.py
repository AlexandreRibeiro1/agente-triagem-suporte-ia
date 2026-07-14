# =============================================================================
# Módulo: Agente de Triagem e Atendimento com IA (v2 — com RAG)
#
# Evolução do agente original. Antes, o agente só classificava a urgência e
# decidia entre "escalar" ou "enfileirar". Agora ele ganha uma capacidade de
# AÇÃO adicional: para chamados normais, ele tenta resolver automaticamente
# consultando uma base de conhecimento (RAG). Só escala para humano quando
# não tem confiança suficiente na resposta.
#
# Ciclo do agente: perceber -> decidir -> agir, agora com 3 desfechos:
#   1) urgente                          -> escalar para plantão
#   2) normal + boa correspondência FAQ -> responder automaticamente
#   3) normal + sem correspondência     -> encaminhar para fila humana
# =============================================================================

from datetime import datetime
import pandas as pd


class AgenteTriagemSuporte:
    def __init__(self, classificador, base_conhecimento):
        self.classificador = classificador
        self.base_conhecimento = base_conhecimento
        self.log_acoes = []

    def perceber_e_classificar(self, texto_chamado):
        """Percepção: usa o modelo de ML para classificar a urgência."""
        return self.classificador.prever(texto_chamado)

    def decidir_e_agir(self, texto_chamado):
        """
        Decisão + ação: define o que fazer com o chamado e já executa,
        incluindo, quando possível, a resposta automática via RAG.
        """
        urgencia = self.perceber_e_classificar(texto_chamado)

        if urgencia == "urgente":
            return {
                "acao_tomada": "🚨 ESCALAR IMEDIATAMENTE para o time de plantão",
                "origem_resposta": "escalonamento",
                "resposta_automatica": None,
                "similaridade": None,
            }

        # Chamado normal: tenta resolver automaticamente via RAG
        resultado_rag = self.base_conhecimento.gerar_resposta_automatica(texto_chamado)

        if resultado_rag is not None:
            return {
                "acao_tomada": "✅ Resolvido automaticamente pelo agente (RAG)",
                "origem_resposta": "rag",
                "resposta_automatica": resultado_rag["resposta_automatica"],
                "similaridade": resultado_rag["similaridade"],
            }

        return {
            "acao_tomada": "📋 Encaminhado para a fila humana (sem correspondência confiável na FAQ)",
            "origem_resposta": "fila_humana",
            "resposta_automatica": None,
            "similaridade": None,
        }

    def agir(self, texto_chamado):
        """Executa o ciclo completo do agente para um chamado e registra o log."""
        urgencia = self.perceber_e_classificar(texto_chamado)
        decisao = self.decidir_e_agir(texto_chamado)

        registro = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chamado": texto_chamado,
            "urgencia_prevista": urgencia,
            "acao_tomada": decisao["acao_tomada"],
            "origem_resposta": decisao["origem_resposta"],
            "similaridade_faq": decisao["similaridade"],
            "resposta_automatica": decisao["resposta_automatica"],
        }
        self.log_acoes.append(registro)
        return registro

    def gerar_relatorio(self):
        """
        Gera um relatório final com estatísticas dos chamados processados,
        incluindo a taxa de resolução automática (KPI-chave para mostrar o
        ganho real de eficiência do agente).
        """
        df = pd.DataFrame(self.log_acoes)
        resumo_urgencia = df["urgencia_prevista"].value_counts()
        resumo_origem = df["origem_resposta"].value_counts()

        total = len(df)
        resolvidos_auto = (df["origem_resposta"] == "rag").sum()
        taxa_resolucao_automatica = resolvidos_auto / total if total else 0

        estatisticas = {
            "total_chamados": total,
            "resumo_urgencia": resumo_urgencia,
            "resumo_origem": resumo_origem,
            "taxa_resolucao_automatica": taxa_resolucao_automatica,
        }
        return df, estatisticas
