import numpy as np

def janela_deslizante(historico, janela=4):
    X, y = [], []
    for i in range(len(historico) - janela):
        X.append(historico[i:i + janela])
        y.append(historico[i + janela])
    return np.array(X), np.array(y)

def acuracia(y_real, y_pred):
    return np.mean(y_real == y_pred) * 100

def normalizar(X_treino, X_teste):
    """Min-max no treino; aplica a mesma escala no teste (sem data leakage).
    Mais robusto que zscore para dados muito desbalanceados (evita outliers de magnitude alta)."""
    xmin = X_treino.min(axis=0)
    xmax = X_treino.max(axis=0)
    rng  = (xmax - xmin) + 1e-8
    return (X_treino - xmin) / rng, (X_teste - xmin) / rng

FORTE_NAO, FRACO_NAO, FRACO_SIM, FORTE_SIM = 0, 1, 2, 3

def prever_2bits(estado):   return 1 if estado >= FRACO_SIM else 0
def atualizar_2bits(estado, tomado):
    return min(estado + 1, FORTE_SIM) if tomado else max(estado - 1, FORTE_NAO)

def rodar_2bits(historico):
    estado, predicoes = FRACO_NAO, []
    for tomado in historico:
        predicoes.append(prever_2bits(estado))
        estado = atualizar_2bits(estado, tomado)
    predicoes = np.array(predicoes)
    return predicoes, acuracia(predicoes, historico)

def sigmoid(z):  return 1 / (1 + np.exp(-z))

def treinar_logistica(X, y, taxa=0.1, epocas=200):
    pesos, vies = np.zeros(X.shape[1]), 0.0
    for _ in range(epocas):
        pred  = sigmoid(X @ pesos + vies)
        erro  = pred - y
        pesos -= taxa * (X.T @ erro) / len(y)
        vies  -= taxa * np.mean(erro)
    return pesos, vies

def prever_logistica(X, pesos, vies):
    return (sigmoid(X @ pesos + vies) >= 0.5).astype(int)

def gini(y):
    if len(y) == 0: return 0
    p = np.mean(y)
    return 2 * p * (1 - p)

def melhor_divisao(X, y):
    melhor_ganho, melhor_feat = -1, 0
    G = gini(y)
    for feat in range(X.shape[1]):
        esq = y[X[:, feat] == 0];  dir_ = y[X[:, feat] == 1]
        if len(esq) == 0 or len(dir_) == 0: continue
        ganho = G - (len(esq)/len(y))*gini(esq) - (len(dir_)/len(y))*gini(dir_)
        if ganho > melhor_ganho:
            melhor_ganho, melhor_feat = ganho, feat
    return melhor_feat

def construir_arvore(X, y, profundidade, max_prof=3, min_amostras=10):
    if (profundidade == max_prof
            or len(np.unique(y)) == 1
            or len(y) < min_amostras):
        return int(np.round(np.mean(y)))
    feat     = melhor_divisao(X, y)
    mask_esq = X[:, feat] == 0
    return {
        "feat":     feat,
        "esquerda": construir_arvore(X[mask_esq],  y[mask_esq],  profundidade+1, max_prof, min_amostras),
        "direita":  construir_arvore(X[~mask_esq], y[~mask_esq], profundidade+1, max_prof, min_amostras),
    }

def prever_um(x, no):
    if isinstance(no, int): return no
    return prever_um(x, no["esquerda"] if x[no["feat"]] == 0 else no["direita"])

def prever_arvore(X, arvore):
    return np.array([prever_um(x, arvore) for x in X])

def eh_linearmente_inseparavel(X, y):
    """
    Detecta se o mesmo vetor x aparece com classes diferentes.
    Nesse caso, nenhum hiperparâmetro salva o Perceptron.
    """
    vistos = {}
    for xi, yi in zip(map(tuple, X), y):
        if xi in vistos and vistos[xi] != yi:
            return True
        vistos[xi] = yi
    return False

def treinar_perceptron(X, y, taxa=0.01, epocas=300):
    p0    = np.clip(np.mean(y), 0.01, 0.99)
    pesos = np.zeros(X.shape[1])
    vies  = np.log(p0 / (1 - p0))   # viés inicial = logit da freq. de 1s
    for _ in range(epocas):
        for xi, yi in zip(X, y):
            pred  = int((xi @ pesos + vies) >= 0)
            erro  = yi - pred
            pesos += np.clip(taxa * erro * xi, -1.0, 1.0)
            vies  += np.clip(taxa * erro,      -1.0, 1.0)
    return pesos, vies

def prever_perceptron(X, pesos, vies):
    return (X @ pesos + vies >= 0).astype(int)

def rodar_experimento(nome, historico, janela=4):
    X, y   = janela_deslizante(historico, janela)
    corte  = int(len(X) * 0.8)
    Xtr, ytr = X[:corte], y[:corte]
    Xte, yte = X[corte:], y[corte:]

    Xtr_n, Xte_n = normalizar(Xtr, Xte)

    # avalia 2-bit sobre o mesmo subconjunto de teste dos modelos ML
    preds_2bit, estado = [], FRACO_NAO
    for tomado in historico:
        preds_2bit.append(prever_2bits(estado))
        estado = atualizar_2bits(estado, tomado)
    preds_2bit = np.array(preds_2bit)
    # X[i] prediz historico[i+janela]; teste começa em X[corte] → historico[corte+janela]
    inicio = corte + janela
    acc_2bits = acuracia(historico[inicio:inicio+len(yte)], preds_2bit[inicio:inicio+len(yte)])

    pesos_log,  vies_log  = treinar_logistica(Xtr_n, ytr)
    arvore                = construir_arvore(Xtr, ytr, profundidade=0)

    inseparavel = eh_linearmente_inseparavel(Xtr, ytr)
    if not inseparavel:
        pesos_perc, vies_perc = treinar_perceptron(Xtr_n, ytr)
        acc_perc = acuracia(yte, prever_perceptron(Xte_n, pesos_perc, vies_perc))
        perc_str = f"{acc_perc:5.1f}%"
        delta_p  = acc_perc - acc_2bits
        perc_delta = f"({'+' if delta_p>=0 else ''}{delta_p:.1f}% vs clássico)"
    else:
        perc_str   = "  N/A"
        perc_delta = "(inseparável: mesmo X → classes 0 e 1)"

    acc_log  = acuracia(yte, prever_logistica(Xte_n, pesos_log,  vies_log))
    acc_arv  = acuracia(yte, prever_arvore(Xte,      arvore))

    def linha(modelo, acc):
        d = acc - acc_2bits
        print(f"  {modelo:<24}  {acc:5.1f}%   ({'+' if d>=0 else ''}{d:.1f}% vs clássico)")

    print(f"\n{'─'*52}")
    print(f"  Padrão: {nome}  ({len(historico)} branches)")
    print(f"{'─'*52}")
    print(f"  {'2-bit clássico':<24}  {acc_2bits:5.1f}%   (baseline)")
    linha("Regressão Logística",  acc_log)
    linha("Árvore de Decisão",    acc_arv)
    print(f"  {'Perceptron':<24}  {perc_str}   {perc_delta}")

np.random.seed(42)
loop      = np.array(([1] * 99 + [0]) * 5)
alternado = np.tile([1, 0], 250)
aleatorio = (np.random.rand(500) < 0.7).astype(int)

print("=" * 52)
print("   BRANCH PREDICTION — 2-bit clássico vs ML")
print("=" * 52)
rodar_experimento("Loop (99% taken)",      loop)
rodar_experimento("Alternado (50/50)",     alternado)
rodar_experimento("Aleatório (70% taken)", aleatorio)
print(f"\n{'='*52}")