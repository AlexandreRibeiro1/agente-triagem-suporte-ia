# =============================================================================
# DESAFIO FINAL - Trilha Especialista em Inteligência Artificial (Alura)
# Projeto: Agente de Triagem Automática de Chamados de Suporte com IA
#
# Objetivo: aplicar Machine Learning + lógica de Agente de IA + Python para
# automatizar um processo real (triagem e priorização de chamados de suporte),
# reduzindo o tempo de resposta a chamados urgentes e organizando o backlog.
#
# Autor: Alexandre (Ale)
# =============================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. BASE DE TREINAMENTO
# Dataset de exemplo com chamados de suporte já rotulados por urgência.
# Em um cenário real, isso viria de um histórico de chamados (CRM, Zendesk,
# planilha de atendimento, etc).
# -----------------------------------------------------------------------------
dados_treino = [
    ("Sistema fora do ar, não conseguimos vender nada", "urgente"),
    ("Erro crítico ao processar pagamento dos clientes", "urgente"),
    ("Aplicativo travando para todos os usuários agora", "urgente"),
    ("Perdemos acesso ao banco de dados de produção", "urgente"),
    ("Cliente não consegue finalizar compra, loja parada", "urgente"),
    ("Servidor caiu e o site está inacessível", "urgente"),
    ("Vazamento de dados foi identificado, ação imediata necessária", "urgente"),
    ("Não conseguimos emitir nenhuma nota fiscal, urgente", "urgente"),
    ("Falha de segurança grave identificada no sistema de login", "urgente"),
    ("Brecha de segurança crítica precisa ser corrigida agora", "urgente"),
    ("Todos os usuários perderam acesso ao sistema simultaneamente", "urgente"),
    ("Ataque hacker em andamento, precisamos agir imediatamente", "urgente"),
    ("Dados de clientes expostos, situação crítica de segurança", "urgente"),
    ("Sistema de pagamento parado, vendas zeradas há uma hora", "urgente"),
    ("Toda a equipe está sem acesso ao sistema neste momento", "urgente"),
    ("Aplicativo não abre para nenhum usuário, produção parada", "urgente"),
    ("Gostaria de saber como alterar minha senha", "normal"),
    ("Dúvida sobre como funciona o relatório mensal", "normal"),
    ("Poderiam me explicar como exportar os dados para Excel?", "normal"),
    ("Sugestão de melhoria para o menu do sistema", "normal"),
    ("Qual o prazo de entrega do meu pedido?", "normal"),
    ("Como faço para atualizar meus dados cadastrais?", "normal"),
    ("Gostaria de um tutorial sobre o novo dashboard", "normal"),
    ("Ótimo atendimento, só queria elogiar a equipe", "normal"),
    ("Pequeno erro visual no botão de login, não impede o uso", "normal"),
    ("Poderia me indicar um material de apoio sobre a plataforma?", "normal"),
    ("O relatório demorou alguns segundos a mais que o normal", "normal"),
    ("Como cancelo minha assinatura no próximo mês?", "normal"),
    ("Não recebi o e-mail de confirmação do meu cadastro", "normal"),
    ("Gostaria de sugerir uma nova funcionalidade no app", "normal"),
    ("Como faço para trocar meu endereço de entrega?", "normal"),
    ("Poderia me explicar como funciona o plano premium?", "normal"),
]

df_treino = pd.DataFrame(dados_treino, columns=["texto", "urgencia"])

# -----------------------------------------------------------------------------
# 2. TREINAMENTO DO MODELO DE MACHINE LEARNING
# Pipeline: TF-IDF (vetorização de texto) + Naive Bayes (classificação)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_treino["texto"], df_treino["urgencia"],
    test_size=0.25, random_state=42, stratify=df_treino["urgencia"]
)

modelo = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classificador", MultinomialNB()),
])

modelo.fit(X_train, y_train)

previsoes_teste = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes_teste)

print("=" * 70)
print("TREINAMENTO DO MODELO")
print("=" * 70)
print(f"Acurácia no conjunto de teste: {acuracia:.0%}\n")
print(classification_report(y_test, previsoes_teste, zero_division=0))


# -----------------------------------------------------------------------------
# 3. AGENTE DE IA
# O agente recebe novos chamados, usa o modelo de ML para classificar a
# urgência e TOMA UMA DECISÃO AUTOMÁTICA sobre o que fazer com cada chamado,
# sem intervenção humana. Isso é o que caracteriza um agente (percepção ->
# decisão -> ação), e não apenas um classificador isolado.
# -----------------------------------------------------------------------------
class AgenteTriagemSuporte:
    def __init__(self, modelo_ml):
        self.modelo = modelo_ml
        self.log_acoes = []

    def perceber_e_classificar(self, texto_chamado):
        """Percepção: usa o modelo de ML para classificar o chamado."""
        return self.modelo.predict([texto_chamado])[0]

    def decidir_acao(self, urgencia):
        """Decisão: define a ação automática com base na urgência."""
        if urgencia == "urgente":
            return "🚨 ESCALAR IMEDIATAMENTE para o time de plantão"
        else:
            return "📋 Adicionar à fila normal de atendimento"

    def agir(self, texto_chamado):
        """Ação: executa o ciclo completo do agente para um chamado."""
        urgencia = self.perceber_e_classificar(texto_chamado)
        acao = self.decidir_acao(urgencia)

        registro = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chamado": texto_chamado,
            "urgencia_prevista": urgencia,
            "acao_tomada": acao,
        }
        self.log_acoes.append(registro)
        return registro

    def gerar_relatorio(self):
        """Gera um relatório final com estatísticas dos chamados processados,
        no mesmo espírito de um painel de KPIs."""
        df = pd.DataFrame(self.log_acoes)
        resumo = df["urgencia_prevista"].value_counts()
        return df, resumo


# -----------------------------------------------------------------------------
# 4. SIMULANDO A CHEGADA DE NOVOS CHAMADOS (dados nunca vistos pelo modelo)
# -----------------------------------------------------------------------------
novos_chamados = [
    "O site caiu para todos os clientes, ninguém consegue comprar",
    "Como faço para trocar meu endereço de entrega?",
    "Identificamos uma falha de segurança grave no login",
    "Gostaria de elogiar a rapidez do suporte de ontem",
    "Não recebi o e-mail de confirmação do pedido",
    "Toda a equipe está sem acesso ao sistema agora",
]

agente = AgenteTriagemSuporte(modelo)

print("\n" + "=" * 70)
print("AGENTE PROCESSANDO NOVOS CHAMADOS")
print("=" * 70)
for chamado in novos_chamados:
    registro = agente.agir(chamado)
    print(f"\n📨 Chamado: {registro['chamado']}")
    print(f"   → Urgência prevista: {registro['urgencia_prevista'].upper()}")
    print(f"   → Ação do agente: {registro['acao_tomada']}")

# -----------------------------------------------------------------------------
# 5. RELATÓRIO FINAL AUTOMATIZADO (saída do processo)
# -----------------------------------------------------------------------------
df_log, resumo = agente.gerar_relatorio()

print("\n" + "=" * 70)
print("RELATÓRIO FINAL - RESUMO DE TRIAGEM")
print("=" * 70)
print(resumo.to_string())
print(f"\nTotal de chamados processados: {len(df_log)}")

caminho_saida = "/home/claude/desafio_final/relatorio_triagem.csv"
df_log.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
print(f"\n✅ Relatório salvo em: {caminho_saida}")
