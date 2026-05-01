import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO


st.set_page_config(
    page_title="Branch Predictor — ML vs Clássico",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e8f0;
}
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

.metric-card {
    background: linear-gradient(135deg, #12122a 0%, #1a1a35 100%);
    border: 1px solid #2a2a50;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #5a5aff; }
.metric-value { font-size: 2.2rem; font-weight: 800; color: #7c7cff; }
.metric-delta-pos { color: #4ecca3; font-size: 0.9rem; }
.metric-delta-neg { color: #ff6b6b; font-size: 0.9rem; }
.metric-label { color: #888; font-size: 0.85rem; margin-top: 4px; }

.insight-box {
    background: #12122a;
    border-left: 3px solid #5a5aff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #b0b0d0;
}
.na-badge {
    background: #2a1a1a;
    border: 1px solid #ff6b6b44;
    color: #ff6b6b;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
.stSelectbox label, .stSlider label, .stFileUploader label { color: #aaa !important; font-size: 0.85rem !important; }
div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; }

.section-title {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5aff;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


def janela_deslizante(historico, janela=4):
    X, y = [], []
    for i in range(len(historico) - janela):
        X.append(historico[i:i + janela])
        y.append(historico[i + janela])
    return np.array(X), np.array(y)

def acuracia(y_real, y_pred):
    return np.mean(y_real == y_pred) * 100

def normalizar(Xtr, Xte):
    xmin = Xtr.min(0); xmax = Xtr.max(0); rng = (xmax - xmin) + 1e-8
    return (Xtr - xmin) / rng, (Xte - xmin) / rng

FORTE_NAO, FRACO_NAO, FRACO_SIM, FORTE_SIM = 0, 1, 2, 3
def prever_2bits(e): return 1 if e >= FRACO_SIM else 0
def atualizar_2bits(e, t): return min(e+1, FORTE_SIM) if t else max(e-1, FORTE_NAO)

def sigmoid(z): return 1 / (1 + np.exp(-z))

def treinar_logistica(X, y, taxa=0.1, epocas=200):
    p, v = np.zeros(X.shape[1]), 0.0
    for _ in range(epocas):
        pr = sigmoid(X @ p + v); e = pr - y
        p -= taxa * (X.T @ e) / len(y); v -= taxa * np.mean(e)
    return p, v

def prever_logistica(X, p, v):
    return (sigmoid(X @ p + v) >= 0.5).astype(int)

def gini(y):
    if len(y) == 0: return 0
    p = np.mean(y); return 2 * p * (1 - p)

def melhor_divisao(X, y):
    mg, mf = -1, 0; G = gini(y)
    for f in range(X.shape[1]):
        e = y[X[:, f]==0]; d = y[X[:, f]==1]
        if len(e)==0 or len(d)==0: continue
        g = G - (len(e)/len(y))*gini(e) - (len(d)/len(y))*gini(d)
        if g > mg: mg, mf = g, f
    return mf

def construir_arvore(X, y, prof, mp=3, ms=10):
    if prof==mp or len(np.unique(y))==1 or len(y)<ms: return int(np.round(np.mean(y)))
    f = melhor_divisao(X, y); m = X[:, f]==0
    return {"feat": f,
            "esquerda": construir_arvore(X[m],  y[m],  prof+1, mp, ms),
            "direita":  construir_arvore(X[~m], y[~m], prof+1, mp, ms)}

def prever_um(x, no):
    if isinstance(no, int): return no
    return prever_um(x, no["esquerda"] if x[no["feat"]]==0 else no["direita"])

def prever_arvore(X, arv):
    return np.array([prever_um(x, arv) for x in X])

def eh_inseparavel(X, y):
    v = {}
    for xi, yi in zip(map(tuple, X), y):
        if xi in v and v[xi] != yi: return True
        v[xi] = yi
    return False

def treinar_perceptron(X, y, taxa=0.01, epocas=300):
    p0 = np.clip(np.mean(y), 0.01, 0.99)
    p, v = np.zeros(X.shape[1]), np.log(p0/(1-p0))
    for _ in range(epocas):
        for xi, yi in zip(X, y):
            pr = int((xi@p+v)>=0); e = yi-pr
            p += np.clip(taxa*e*xi, -1, 1); v += np.clip(taxa*e, -1, 1)
    return p, v

def prever_perceptron(X, p, v):
    return (X @ p + v >= 0).astype(int)

@st.cache_data
def rodar_experimento(historico_tuple, janela=4):
    historico = np.array(historico_tuple)
    X, y = janela_deslizante(historico, janela)
    corte = int(len(X) * 0.8)
    Xtr, ytr = X[:corte], y[:corte]
    Xte, yte = X[corte:], y[corte:]
    Xtr_n, Xte_n = normalizar(Xtr, Xte)

    # 2-bit no mesmo slice de teste
    preds_2bit, estado = [], FRACO_NAO
    for t in historico:
        preds_2bit.append(prever_2bits(estado))
        estado = atualizar_2bits(estado, t)
    preds_2bit = np.array(preds_2bit)
    inicio = corte + janela
    acc_2bits = acuracia(historico[inicio:inicio+len(yte)], preds_2bit[inicio:inicio+len(yte)])

    plog, vlog = treinar_logistica(Xtr_n, ytr)
    arv = construir_arvore(Xtr, ytr, 0)
    acc_log = acuracia(yte, prever_logistica(Xte_n, plog, vlog))
    acc_arv = acuracia(yte, prever_arvore(Xte, arv))

    insep = eh_inseparavel(Xtr, ytr)
    if not insep:
        pp, vp = treinar_perceptron(Xtr_n, ytr)
        acc_perc = acuracia(yte, prever_perceptron(Xte_n, pp, vp))
    else:
        acc_perc = None

    # histórico de estados 2-bit para visualização
    estados, preds_hist = [], []
    estado = FRACO_NAO
    for t in historico:
        preds_hist.append(prever_2bits(estado))
        estados.append(estado)
        estado = atualizar_2bits(estado, t)

    return {
        "acc_2bits": acc_2bits,
        "acc_log":   acc_log,
        "acc_arv":   acc_arv,
        "acc_perc":  acc_perc,
        "insep":     insep,
        "estados":   estados,
        "preds_2bit_hist": preds_hist,
        "historico": historico.tolist(),
        "yte":       yte.tolist(),
        "n_treino":  len(ytr),
        "n_teste":   len(yte),
    }


with st.sidebar:
    st.markdown("## 🧠 Branch Predictor")
    st.markdown("---")

    st.markdown('<p class="section-title">Fonte dos dados</p>', unsafe_allow_html=True)
    fonte = st.radio("", ["Padrões pré-definidos", "Upload CSV"], label_visibility="collapsed")

    if fonte == "Padrões pré-definidos":
        np.random.seed(42)
        padroes_disponiveis = {
            "Loop (99% taken)":      ([1]*99 + [0]) * 5,
            "Alternado (50/50)":     list(np.tile([1,0], 250)),
            "Aleatório (70% taken)": list((np.random.rand(500) < 0.7).astype(int)),
            "Burst (75% clusters)":  list(np.repeat(np.random.choice([0,1], 100, p=[0.25,0.75]), 5)[:500]),
        }
        padrao_sel = st.selectbox("Padrão", list(padroes_disponiveis.keys()))
        historico_raw = padroes_disponiveis[padrao_sel]
        nome_padrao = padrao_sel
    else:
        uploaded = st.file_uploader("CSV com coluna 'taken' (0/1)", type="csv")
        if uploaded:
            df_up = pd.read_csv(uploaded)
            col = st.selectbox("Coluna", df_up.columns.tolist())
            historico_raw = df_up[col].astype(int).tolist()
            nome_padrao = f"CSV: {col}"
        else:
            st.info("Aguardando arquivo...")
            historico_raw = ([1]*99 + [0]) * 5
            nome_padrao = "Loop (99% taken)"

    st.markdown("---")
    st.markdown('<p class="section-title">Configuração</p>', unsafe_allow_html=True)
    janela = st.slider("Janela (branches anteriores)", 2, 16, 4)

    st.markdown("---")
    st.caption(f"**{len(historico_raw)}** branches  ·  janela={janela}")
    st.caption("Treino: 80%  ·  Teste: 20%")

historico = np.array(historico_raw)
res = rodar_experimento(tuple(historico_raw), janela)

st.markdown(f"# Branch Prediction")
st.markdown(f"### `{nome_padrao}` · {len(historico)} branches · janela={janela}")
st.markdown("---")

modelos = [
    ("2-bit Clássico", res["acc_2bits"], None, "baseline"),
    ("Regressão Logística", res["acc_log"], res["acc_2bits"], "ML"),
    ("Árvore de Decisão", res["acc_arv"], res["acc_2bits"], "ML"),
    ("Perceptron", res["acc_perc"], res["acc_2bits"], "ML"),
]

cols = st.columns(4)
for col, (nome, acc, base, tag) in zip(cols, modelos):
    with col:
        if acc is None:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;color:#ff6b6b">N/A</div>
                <div class="metric-label">{nome}</div>
                <div class="na-badge" style="margin-top:8px">linearmente inseparável</div>
            </div>""", unsafe_allow_html=True)
        else:
            delta = acc - base if base is not None else 0
            delta_html = ""
            if base is not None:
                sinal = "+" if delta >= 0 else ""
                cls = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
                delta_html = f'<div class="{cls}">{sinal}{delta:.1f}% vs clássico</div>'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{acc:.1f}%</div>
                <div class="metric-label">{nome}</div>
                {delta_html}
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<p class="section-title">Acurácia por modelo</p>', unsafe_allow_html=True)

    nomes_graf = ["2-bit Clássico", "Reg. Logística", "Árvore Decisão", "Perceptron"]
    accs_graf  = [res["acc_2bits"], res["acc_log"], res["acc_arv"],
                  res["acc_perc"] if res["acc_perc"] is not None else 0]
    cores = ["#5a5aff", "#4ecca3", "#f7c59f", "#ff6b6b" if res["acc_perc"] is None else "#a78bfa"]
    textos = [f"{a:.1f}%" if (i!=3 or res["acc_perc"] is not None) else "N/A"
              for i, a in enumerate(accs_graf)]

    fig_bar = go.Figure(go.Bar(
        x=nomes_graf, y=accs_graf,
        marker_color=cores,
        text=textos, textposition="outside",
        textfont=dict(family="JetBrains Mono", size=13, color="#e8e8f0"),
    ))
    fig_bar.update_layout(
        plot_bgcolor="#12122a", paper_bgcolor="#0a0a0f",
        font=dict(color="#e8e8f0", family="Syne"),
        yaxis=dict(range=[0, 115], gridcolor="#1e1e30", zeroline=False),
        xaxis=dict(gridcolor="#1e1e30"),
        margin=dict(t=20, b=10, l=10, r=10),
        height=300,
        showlegend=False,
    )
    # linha do baseline
    fig_bar.add_hline(y=res["acc_2bits"], line_dash="dot", line_color="rgba(90,90,255,0.3)",
                      annotation_text="baseline", annotation_font_color="#5a5aff")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown('<p class="section-title">Distribuição do padrão</p>', unsafe_allow_html=True)
    taken_pct  = np.mean(historico) * 100
    ntaken_pct = 100 - taken_pct
    fig_pie = go.Figure(go.Pie(
        labels=["Taken (1)", "Not Taken (0)"],
        values=[taken_pct, ntaken_pct],
        hole=0.6,
        marker_colors=["#4ecca3", "#ff6b6b"],
        textfont=dict(family="JetBrains Mono", size=12),
    ))
    fig_pie.update_layout(
        plot_bgcolor="#12122a", paper_bgcolor="#0a0a0f",
        font=dict(color="#e8e8f0", family="Syne"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        showlegend=True,
        legend=dict(font=dict(size=11)),
        annotations=[dict(text=f"{taken_pct:.0f}%<br>taken",
                          font_size=16, font_color="#4ecca3",
                          showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown('<p class="section-title">Evolução do estado 2-bit ao longo do tempo</p>', unsafe_allow_html=True)

n_mostrar = st.slider("Branches a visualizar", 20, min(500, len(historico)), min(100, len(historico)))
estados_vis = res["estados"][:n_mostrar]
hist_vis    = res["historico"][:n_mostrar]
preds_vis   = res["preds_2bit_hist"][:n_mostrar]
acertos_vis = [int(p == t) for p, t in zip(preds_vis, hist_vis)]

state_names = {0: "SNT", 1: "WNT", 2: "WT", 3: "ST"}
state_colors = {0: "#ff6b6b", 1: "#f7c59f", 2: "#a78bfa", 3: "#4ecca3"}

fig_state = go.Figure()
fig_state.add_trace(go.Scatter(
    x=list(range(n_mostrar)), y=estados_vis,
    mode="lines+markers",
    line=dict(color="#5a5aff", width=1.5),
    marker=dict(
        color=[state_colors[s] for s in estados_vis],
        size=6,
        symbol=["circle" if a else "x" for a in acertos_vis],
    ),
    name="Estado 2-bit",
    hovertemplate="Branch %{x}<br>Estado: %{text}<br>Acerto: %{customdata}",
    text=[state_names[s] for s in estados_vis],
    customdata=["✓" if a else "✗" for a in acertos_vis],
))
fig_state.add_trace(go.Scatter(
    x=list(range(n_mostrar)), y=hist_vis,
    mode="lines", line=dict(color="rgba(255,255,255,0.13)", width=1, dash="dot"),
    name="Branch real (0/1)", yaxis="y2",
))
fig_state.update_layout(
    plot_bgcolor="#12122a", paper_bgcolor="#0a0a0f",
    font=dict(color="#e8e8f0", family="Syne"),
    yaxis=dict(tickvals=[0,1,2,3], ticktext=["SNT","WNT","WT","ST"],
               gridcolor="#1e1e30", title="Estado"),
    yaxis2=dict(overlaying="y", side="right", showgrid=False,
                tickvals=[0,1], ticktext=["NT","T"], title="Branch"),
    xaxis=dict(gridcolor="#1e1e30", title="Branch #"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10, l=10, r=10),
    height=280,
)
st.plotly_chart(fig_state, use_container_width=True)

st.markdown("---")
st.markdown('<p class="section-title">Análise automática</p>', unsafe_allow_html=True)

taken_pct = np.mean(historico) * 100
insights = []

if taken_pct > 90:
    insights.append("🔵 Padrão altamente enviesado — 2-bit clássico é quase ótimo por design.")
if taken_pct < 55 and taken_pct > 45:
    insights.append("🟡 Padrão alternado ou equilibrado — 2-bit sofre, modelos com memória de sequência dominam.")
if res["acc_log"] > res["acc_2bits"] + 5:
    insights.append(f"🟢 Regressão Logística supera o clássico em +{res['acc_log']-res['acc_2bits']:.1f}% — há correlação temporal explorável.")
if res["acc_arv"] > res["acc_2bits"] + 5:
    insights.append(f"🟢 Árvore de Decisão supera o clássico em +{res['acc_arv']-res['acc_2bits']:.1f}% — há divisões lógicas no padrão.")
if res["insep"]:
    insights.append("🔴 Dados linearmente inseparáveis detectados — Perceptron não converge por construção matemática.")
if res["acc_perc"] is not None and res["acc_perc"] < 50:
    insights.append("🔴 Perceptron abaixo de 50% — pior que chute aleatório. Dados sem separabilidade linear.")
if abs(taken_pct - 70) < 5 and res["acc_log"] < 70:
    insights.append("⚪ Acurácia próxima ao teto teórico — padrão sem correlação temporal explorável.")

if not insights:
    insights.append("✅ Todos os modelos performando conforme esperado para este padrão.")

for ins in insights:
    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="section-title">Tabela comparativa</p>', unsafe_allow_html=True)

rows = []
for nome, acc, base in [
    ("2-bit Clássico",     res["acc_2bits"], None),
    ("Regressão Logística",res["acc_log"],   res["acc_2bits"]),
    ("Árvore de Decisão",  res["acc_arv"],   res["acc_2bits"]),
    ("Perceptron",         res["acc_perc"],  res["acc_2bits"]),
]:
    if acc is None:
        rows.append({"Modelo": nome, "Acurácia": "N/A", "Δ vs 2-bit": "inseparável", "Recomendado para": "—"})
    else:
        delta = (acc - base) if base is not None else 0
        rows.append({
            "Modelo": nome,
            "Acurácia": f"{acc:.1f}%",
            "Δ vs 2-bit": ("—" if base is None else f"{'+' if delta>=0 else ''}{delta:.1f}%"),
            "Recomendado para": {
                "2-bit Clássico": "loops, padrões estáveis",
                "Regressão Logística": "qualquer padrão com correlação",
                "Árvore de Decisão": "padrões com regras booleanas",
                "Perceptron": "padrões linearmente separáveis",
            }.get(nome, "—")
        })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Branch Prediction · 2-bit Clássico vs ML · Implementado do zero com NumPy")