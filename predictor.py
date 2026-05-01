import numpy as np

SNT, WNT, WT, ST = 0, 1, 2, 3
STATE_NAMES = {SNT: "SNT", WNT: "WNT", WT: "WT", ST: "ST"}

def predict(state):
    return 1 if state >= WT else 0

def update(state, taken):
    if taken:
        return min(state + 1, ST)
    else:
        return max(state - 1, SNT)

def run_predictor(history):
    state = WNT
    predictions = []

    for taken in history:
        pred = predict(state)
        predictions.append(pred)
        state = update(state, taken)

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == history) * 100
    return predictions, accuracy

np.random.seed(42)

loop_pattern = np.array(([1] * 99 + [0]) * 5)
alternating = np.tile([1, 0], 250)
random_biased = (np.random.rand(500) < 0.7).astype(int)

padroes = {
    "Loop (99% taken)":     loop_pattern,
    "Alternado (50/50)":    alternating,
    "Aleatório (70% taken)": random_biased,
}

print("=" * 50)
print("  PREDITOR DE BRANCH — 2 bits")
print("=" * 50)

for nome, history in padroes.items():
    preds, acc = run_predictor(history)
    erros = np.sum(preds != history)
    print(f"\n{nome}")
    print(f"  Branches : {len(history)}")
    print(f"  Flushes  : {erros}")
    print(f"  Acurácia : {acc:.1f}%")

print("\n" + "=" * 50)
print("Histórico salvo em history.csv para o modelo ML")
np.savetxt("history.csv", random_biased, fmt="%d", header="taken", comments="")