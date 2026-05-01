<img width="298" height="300" alt="Captura de Tela 2026-05-01 às 19 07 54" src="https://github.com/user-attachments/assets/812e146f-05fd-4369-9492-9f4f2eff2d9c" />
<img width="1437" height="805" alt="Captura de Tela 2026-05-01 às 19 07 40" src="https://github.com/user-attachments/assets/98af9784-2162-43e0-92b2-79ef37b105c3" />

**[Live Demo →](https://cpu-branch-predictor-4yets6evdnqkcrfnyfbdan.streamlit.app/)**

# cpu-branch-predictor

> Interactive dashboard comparing classical 2-bit branch prediction against ML models implemented from scratch — no scikit-learn, no PyTorch, no ML libraries.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy_only-no_sklearn-013243?style=flat-square&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## ◈ Overview

Modern CPUs rely on branch predictors to speculatively execute instructions before a conditional jump is resolved. A misprediction flushes the pipeline — typically 10–20 wasted cycles.

This project benchmarks the classical **2-bit saturating counter** (the industry baseline since the 1980s) against three ML models trained on the same branch history, using a sliding window of N prior outcomes as features.

All models — logistic regression, decision tree, and perceptron — are implemented entirely in **NumPy**. No scikit-learn, no ML frameworks.

---

## ◈ Models

| Model | Implementation |
|---|---|
| 2-bit Saturating Counter | 4-state FSM (SNT → WNT → WT → ST) |
| Logistic Regression | Gradient descent + sigmoid, from scratch |
| Decision Tree | Gini impurity, configurable depth, from scratch |
| Perceptron | Online learning with logit bias initialization |

### Design decisions worth noting

**Normalization without data leakage** — min-max scaling is fit exclusively on the training split; the same parameters are applied to the test set. Standard practice, but easy to get wrong.

**Gini impurity** — implemented as `2·p·(1−p)` (the correct binary form), not the multiclass approximation sometimes seen in textbook implementations.

**Perceptron bias initialization** — initialized to `logit(mean(y))` rather than zero, which accelerates convergence on imbalanced branch patterns (e.g. 99% taken loops).

**Linear inseparability detection** — before training the perceptron, the dataset is scanned for conflicting labels on identical feature vectors. If detected, training is skipped and the result is reported as N/A rather than returning a meaningless accuracy.

**`min_samples` on tree leaves** — set to 10 by default to prevent overfitting on small test splits.

---

## ◈ Branch Patterns

Three canonical patterns are included:

```
Loop (99% taken)      →  [1,1,1,...,1,0] × 5       500 branches
Alternating (50/50)   →  [1,0,1,0,...]              500 branches
Random (70% taken)    →  Bernoulli(p=0.7)           500 branches
```

Custom patterns can be loaded via CSV upload in the dashboard.

---

## ◈ Project Structure

```
cpu-branch-predictor/
├── app.py               # Streamlit dashboard (Plotly visualizations)
├── ml_predictor.py      # CLI: runs all models, prints comparison table
├── predictor.py         # Standalone 2-bit predictor, exports history CSV
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── history.csv      # Sample branch history (Bernoulli 70%)
```

---

## ◈ Getting Started

```bash
git clone https://github.com/your-username/cpu-branch-predictor
cd cpu-branch-predictor
pip install -r requirements.txt
```

**Run the dashboard:**
```bash
streamlit run app.py
```

**Run CLI comparison:**
```bash
python ml_predictor.py
```

**Run standalone 2-bit predictor:**
```bash
python predictor.py
```

---

## ◈ Requirements

```
streamlit
numpy
pandas
plotly
```

No machine learning libraries. NumPy is used strictly for array operations and random number generation.

---

## ◈ Results (sample)

| Pattern | 2-bit | Log. Regression | Decision Tree | Perceptron |
|---|---|---|---|---|
| Loop (99% taken) | ~99% | ~99% | ~99% | ~99% |
| Alternating (50/50) | ~50% | ~99% | ~99% | N/A |
| Random (70% taken) | ~70% | ~70% | ~70% | ~70% |

The alternating pattern is where ML models outperform the classical predictor by a wide margin — the 2-bit FSM thrashes between states, while logistic regression and the decision tree learn the [1,0] sequence directly from the sliding window features.

---

## ◈ License

MIT

## Author

**Ghabriel Calado**
Computer Science Student | Python & AI

[GitHub](https://github.com/ghcalado) · [LinkedIn](https://www.linkedin.com/in/ghabriel-calado-7132a33b6/)
