# 🔍 Nitaqat Compliance Root Cause Analyzer
### Diagnostic Analytics · Vision 2030 Portfolio Series · App 2 of 4

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Domain](https://img.shields.io/badge/Domain-KSA%20Labor%20Policy-gold)
![Analytics](https://img.shields.io/badge/Analytics-Diagnostic-purple)

---

## 🎯 Problem This Solves

Descriptive dashboards tell you *what* is happening. This app tells you **why**.

Saudi companies failing Nitaqat compliance don't fail randomly — they fail for identifiable, addressable root causes: uncompetitive Saudi salaries, front-loaded hiring before audit cycles, overqualified Saudis placed in low-skill roles, structural gender gaps, and overreliance on temporary contracts.

This tool decomposes every non-compliant company into a **5-dimension root cause score**, enabling HR directors and HRSD analysts to prioritise interventions with evidence rather than assumption.

---

## 🧠 Analytical Framework

### The 5 Root Causes Measured

| Root Cause | Logic | Weight |
|---|---|---|
| **Salary Uncompetitiveness** | Saudi avg salary vs sector benchmark | 30% |
| **Low Recent Saudi Hiring** | % of last-2-year hires that are Saudi | 25% |
| **Education Mismatch** | Bachelor+ Saudis in low-skill roles | 20% |
| **Gender Participation Gap** | Female % vs Vision 2030 target | 15% |
| **Contract Overreliance** | % temporary contracts (Saudis avoid instability) | 10% |

Each dimension is scored 0–100. A **Composite Risk Score** is computed via weighted sum.

---

## 📊 Visualisations

| Chart | Insight |
|---|---|
| **Root Cause Radar** | Spider chart per sector — which cause dominates where |
| **Intensity Heatmap** | Sectors × Root Causes matrix — quick pattern identification |
| **Gap Bar Chart** | Actual Saudization % vs Nitaqat target, sector by sector |
| **Salary Premium Chart** | Saudi vs non-Saudi pay gap by sector |
| **Hiring Seasonality** | Monthly Saudi hiring pattern — reveals reactive compliance |
| **Overqualification Bar** | Saudis with degrees in low-skill roles — talent misallocation |
| **Risk Scatter Matrix** | Compliance gap vs root cause score, bubble = headcount |
| **Priority Risk Cards** | Top 10 at-risk companies with dominant root cause labelled |
| **Analytical Findings Box** | Auto-generated policy insight paragraph |

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/saudi-workforce-diagnostic.git
cd saudi-workforce-diagnostic

# Copy the CSV from App 1, or re-generate it
cp ../saudi-workforce-descriptive/saudi_workforce_data.csv .
# OR generate fresh:
python ../saudi-workforce-descriptive/generate_data.py

pip install -r requirements.txt
streamlit run app.py
```

### Google Colab
```python
!pip install streamlit pyngrok -q
from pyngrok import ngrok
import subprocess, time
proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(3)
print(ngrok.connect(8501).public_url)
```

---

## 📁 File Structure

```
saudi-workforce-diagnostic/
├── app.py                      # Diagnostic Streamlit dashboard
├── saudi_workforce_data.csv    # Shared dataset (from App 1)
├── requirements.txt
└── README.md
```

---

## 🔮 The Analytics Series

| App | Type | Question Answered |
|---|---|---|
| App 1 ✅ | Descriptive | *What* does the workforce look like? |
| **App 2 ✅** | **Diagnostic** | ***Why* are companies failing Nitaqat?** |
| App 3 | Predictive | *Will* a company breach compliance next year? (ML) |
| App 4 | Prescriptive | *What* hiring plan optimises compliance + cost? (LP) |

---

## 💡 Why This Stands Out

Most diagnostic analytics projects in portfolios use generic retail churn or web traffic data. This project:

- Addresses a **live regulatory compliance problem** worth billions in penalties
- Implements a **principled, weighted scoring framework** (not just charts)
- Generates **actionable, specific policy recommendations** from the data
- Is directly relevant to **HRSD, Vision 2030 offices, and GCC HR consultancies**

---

*Data is entirely synthetic, calibrated to GASTAT and HRSD published statistics. No real company or individual data is used.*
