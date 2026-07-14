# =============================================================================
# Módulo: Classificador de Urgência
# Pipeline TF-IDF + Multinomial Naive Bayes que classifica um chamado de
# suporte como "urgente" ou "normal".
#
# Esta é a mesma lógica do projeto original (desafio final da Trilha Alura),
# agora encapsulada em uma classe reutilizável para facilitar a integração
# com o módulo de RAG.
# =============================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Dataset de exemplo com chamados de suporte já rotulados por urgência.
# Em um cenário real, isso viria de um histórico de chamados (CRM, Zendesk,
# planilha de atendimento, etc).
DADOS_TREINO = [
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


class ClassificadorUrgencia:
    """Encapsula o pipeline TF-IDF + Naive Bayes de classificação de urgência."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classificador", MultinomialNB()),
        ])
        self._treinado = False

    def treinar(self, dados=None, test_size=0.25, random_state=42, verbose=True):
        """Treina o modelo e imprime as métricas de avaliação."""
        dados = dados or DADOS_TREINO
        df = pd.DataFrame(dados, columns=["texto", "urgencia"])

        X_train, X_test, y_train, y_test = train_test_split(
            df["texto"], df["urgencia"],
            test_size=test_size, random_state=random_state, stratify=df["urgencia"],
        )

        self.pipeline.fit(X_train, y_train)
        self._treinado = True

        previsoes_teste = self.pipeline.predict(X_test)
        acuracia = accuracy_score(y_test, previsoes_teste)

        if verbose:
            print("=" * 70)
            print("TREINAMENTO DO CLASSIFICADOR DE URGÊNCIA")
            print("=" * 70)
            print(f"Acurácia no conjunto de teste: {acuracia:.0%}\n")
            print(classification_report(y_test, previsoes_teste, zero_division=0))

        return acuracia

    def prever(self, texto_chamado):
        """Retorna a urgência prevista ('urgente' ou 'normal') para um chamado."""
        if not self._treinado:
            raise RuntimeError("Chame treinar() antes de prever().")
        return self.pipeline.predict([texto_chamado])[0]
