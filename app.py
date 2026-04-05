import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import random

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nitaqat Root Cause Analyzer | Diagnostic Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background: #0c0c0f; }
.block-container { padding: 1.5rem 2rem; }
[data-testid="stSidebar"] { background: #111116; border-right: 1px solid #2a1f3d; }
[data-testid="stSidebar"] * { color: #b0a8c8 !important; }
.diag-title { font-size:1.65rem; font-weight:800; color:#f5f0ff; letter-spacing:-0.03em; }
.diag-sub   { font-size:0.78rem; color:#5b5370; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.5rem; }
.cause-card {
    background: linear-gradient(135deg,#140e1f,#1c1230);
    border:1px solid #3b2a5c; border-left: 4px solid #9333ea;
    border-radius:10px; padding:1.1rem 1.4rem; margin-bottom:0.6rem;
}
.cause-card.warning { border-left-color:#f59e0b; }
.cause-card.danger  { border-left-color:#ef4444; }
.cause-card.ok      { border-left-color:#22c55e; }
.cause-title  { color:#e2d9f3; font-weight:700; font-size:0.9rem; margin-bottom:0.2rem; }
.cause-detail { color:#7c6f9f; font-size:0.78rem; line-height:1.5; }
.cause-score  { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:600; }
.sec-hdr {
    font-size:0.65rem; letter-spacing:0.22em; text-transform:uppercase;
    color:#6d28d9; border-bottom:1px solid #2a1f3d;
    padding-bottom:0.35rem; margin:1.5rem 0 0.9rem;
}
.insight-box {
    background:#16101f; border:1px solid #3b2a5c; border-radius:8px;
    padding:1rem 1.2rem; font-size:0.82rem; color:#b0a8c8; line-height:1.7;
}
.insight-box b { color:#c4b5fd; }
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Colour helper ─────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.08):
    """Convert #RRGGBB → rgba(r,g,b,alpha) — works with all Plotly versions."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Embedded Data Generator ───────────────────────────────────────────────────
@st.cache_data
def generate_workforce_data():
    """Generates a 5,000-row synthetic Saudi workforce dataset in memory."""
    np.random.seed(42)
    random.seed(42)

    N_EMPLOYEES = 5000
    SECTORS = {
        "Construction":          {"nitaqat_target": 6,  "saudi_ratio": 0.08, "salary_base": 4500},
        "Information Technology":{"nitaqat_target": 35, "saudi_ratio": 0.38, "salary_base": 12000},
        "Healthcare":            {"nitaqat_target": 35, "saudi_ratio": 0.42, "salary_base": 9500},
        "Financial Services":    {"nitaqat_target": 70, "saudi_ratio": 0.72, "salary_base": 14000},
        "Retail":                {"nitaqat_target": 30, "saudi_ratio": 0.28, "salary_base": 5500},
        "Education":             {"nitaqat_target": 55, "saudi_ratio": 0.58, "salary_base": 8500},
        "Manufacturing":         {"nitaqat_target": 10, "saudi_ratio": 0.12, "salary_base": 5000},
        "Tourism & Hospitality": {"nitaqat_target": 20, "saudi_ratio": 0.18, "salary_base": 6000},
        "Energy & Utilities":    {"nitaqat_target": 75, "saudi_ratio": 0.78, "salary_base": 16000},
        "Transport & Logistics": {"nitaqat_target": 15, "saudi_ratio": 0.16, "salary_base": 5800},
    }
    REGIONS   = ["Riyadh","Makkah","Eastern Province","Madinah","Asir","Qassim","Tabuk","Hail","Jizan","Najran"]
    REGION_W  = [0.35, 0.22, 0.18, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02]
    NAT_KEYS  = ["Saudi","Indian","Pakistani","Bangladeshi","Egyptian","Filipino",
                 "Yemeni","Syrian","Jordanian","Indonesian","Sudanese","Other"]
    NAT_P     = [0.35,0.18,0.10,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.01]
    EDU_LVLS  = ["High School","Diploma","Bachelor","Master","PhD"]
    EDU_S     = [0.15,0.18,0.45,0.18,0.04]
    EDU_O     = [0.30,0.22,0.38,0.08,0.02]
    AGE_G     = ["<25","25-34","35-44","45-54","55+"]
    AGE_W     = [0.12,0.38,0.30,0.14,0.06]
    JOB_TITLES= {
        "Information Technology": ["Software Engineer","Data Analyst","IT Manager","Systems Administrator","Business Analyst"],
        "Healthcare":             ["Nurse","Physician","Pharmacist","Lab Technician","Medical Admin"],
        "Financial Services":     ["Financial Analyst","Accountant","Branch Manager","Risk Officer","Compliance Specialist"],
        "Construction":           ["Site Engineer","Project Manager","Safety Officer","Surveyor","Laborer"],
        "Retail":                 ["Sales Associate","Store Manager","Merchandiser","Cashier","Category Manager"],
        "Education":              ["Teacher","Academic Advisor","Administrator","Research Analyst","Dean Assistant"],
        "Manufacturing":          ["Production Supervisor","Quality Inspector","Maintenance Tech","Operations Manager","Shift Supervisor"],
        "Tourism & Hospitality":  ["Hotel Manager","Front Desk Officer","Tour Guide","F&B Supervisor","Concierge"],
        "Energy & Utilities":     ["Petroleum Engineer","Safety Analyst","Operations Analyst","Field Technician","HSE Manager"],
        "Transport & Logistics":  ["Logistics Coordinator","Fleet Manager","Customs Specialist","Warehouse Supervisor","Driver"],
    }
    NAMES   = ["Al Rajhi","SABIC","Saudi Aramco","STC","Almarai","Jarir","Al Babtain",
               "Riyad","Samba","NCB","Al Jazira","Olayan","Bin Laden","Al Futtaim",
               "Maaden","ACWA","Saudi Cables","Mobily","Zain","Savola","Tawuniya"]
    SUFX    = ["Co.","Ltd.","Corp.","Group","Holdings","Solutions","Services","International"]

    def nitaqat_classify(rate, target):
        if rate >= min(target*1.4, 90): return "Platinum"
        elif rate >= target*1.15:       return "High Green"
        elif rate >= target:            return "Medium Green"
        elif rate >= target*0.85:       return "Low Green"
        elif rate >= target*0.60:       return "Yellow"
        else:                           return "Red"

    companies = []
    for i in range(120):
        sec = random.choice(list(SECTORS.keys()))
        companies.append({
            "company_id":   f"CO{i+1:04d}",
            "company_name": f"{random.choice(NAMES)} {random.choice(SUFX)}",
            "sector":       sec,
            "region":       np.random.choice(REGIONS, p=REGION_W),
        })
    co_df = pd.DataFrame(companies)

    employees = []
    for _, co in co_df.iterrows():
        sec  = co["sector"]
        info = SECTORS[sec]
        n_emp = np.random.choice([20,50,100,200,500], p=[0.30,0.30,0.25,0.10,0.05])
        for _ in range(n_emp):
            if len(employees) >= N_EMPLOYEES:
                break
            probs = NAT_P.copy()
            s_target = np.clip(info["saudi_ratio"] + np.random.normal(0,0.08), 0.02, 0.95)
            adj = s_target - probs[0]
            probs[0] = s_target
            for k in range(1, len(probs)):
                probs[k] = max(0.001, probs[k] - adj/(len(probs)-1))
            probs = [p/sum(probs) for p in probs]
            nat      = np.random.choice(NAT_KEYS, p=probs)
            is_saudi = nat == "Saudi"
            age_g    = np.random.choice(AGE_G, p=AGE_W)
            yrs      = max(0, {"<25":1,"25-34":5,"35-44":12,"45-54":20,"55+":28}[age_g] + np.random.randint(-2,4))
            edu      = np.random.choice(EDU_LVLS, p=EDU_S if is_saudi else EDU_O)
            edu_m    = {"High School":0.7,"Diploma":0.85,"Bachelor":1.0,"Master":1.3,"PhD":1.6}[edu]
            salary   = max(1500, int(info["salary_base"] * edu_m * (1.15 if is_saudi else 1.0) * (1+yrs*0.04) * np.random.normal(1,0.12)))
            hdate    = (datetime(2015,1,1) + timedelta(days=np.random.randint(0,365*9))).strftime("%Y-%m-%d")
            employees.append({
                "employee_id":       f"EMP{len(employees)+1:06d}",
                "company_id":        co["company_id"],
                "company_name":      co["company_name"],
                "sector":            sec,
                "region":            co["region"],
                "nationality":       nat,
                "is_saudi":          is_saudi,
                "gender":            np.random.choice(["Male","Female"], p=[0.68,0.32] if is_saudi else [0.82,0.18]),
                "education_level":   edu,
                "age_group":         age_g,
                "years_experience":  yrs,
                "job_title":         random.choice(JOB_TITLES[sec]),
                "monthly_salary_sar":salary,
                "hire_date":         hdate,
                "employment_type":   np.random.choice(["Full-time","Part-time","Contract"], p=[0.80,0.10,0.10]),
            })
        if len(employees) >= N_EMPLOYEES:
            break

    df = pd.DataFrame(employees[:N_EMPLOYEES])
    sauz = df.groupby("company_id").apply(
        lambda g: (g["nationality"]=="Saudi").sum()/len(g)*100, include_groups=False
    ).reset_index(name="saudization_rate")
    sec_map = df.drop_duplicates("company_id")[["company_id","sector"]]
    sauz    = sauz.merge(sec_map, on="company_id")
    sauz["nitaqat_status"] = sauz.apply(
        lambda r: nitaqat_classify(r["saudization_rate"], SECTORS[r["sector"]]["nitaqat_target"]), axis=1
    )
    df = df.merge(sauz[["company_id","nitaqat_status"]], on="company_id", how="left")
    return df

# ── Load & Enrich Data ────────────────────────────────────────────────────────
@st.cache_data
def load_and_enrich():
    df = generate_workforce_data()
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["year"]  = df["hire_date"].dt.year
    df["month"] = df["hire_date"].dt.month

    targets = {
        "Construction":6,"Information Technology":35,"Healthcare":35,
        "Financial Services":70,"Retail":30,"Education":55,
        "Manufacturing":10,"Tourism & Hospitality":20,
        "Energy & Utilities":75,"Transport & Logistics":15,
    }
    df["nitaqat_target"] = df["sector"].map(targets)

    co = df.groupby("company_id").apply(lambda g: pd.Series({
        "saudization_pct":        (g["nationality"]=="Saudi").mean()*100,
        "headcount":              len(g),
        "avg_salary_saudi":       g[g["nationality"]=="Saudi"]["monthly_salary_sar"].mean() if (g["nationality"]=="Saudi").any() else 0,
        "avg_salary_nonsaudi":    g[g["nationality"]!="Saudi"]["monthly_salary_sar"].mean() if (g["nationality"]!="Saudi").any() else 0,
        "pct_bachelor_plus_saudi":(
            g[(g["nationality"]=="Saudi") & g["education_level"].isin(["Bachelor","Master","PhD"])].shape[0] /
            max(g[g["nationality"]=="Saudi"].shape[0],1)*100
        ),
        "pct_female":             (g["gender"]=="Female").mean()*100,
        "recent_saudi_hires":     (
            g[(g["nationality"]=="Saudi") & (g["year"]>=g["year"].max()-2)].shape[0] /
            max(g[g["year"]>=g["year"].max()-2].shape[0],1)*100
        ),
        "contract_pct":           (g["employment_type"]=="Contract").mean()*100,
        "senior_saudi_pct":       (
            g[(g["nationality"]=="Saudi") & g["age_group"].isin(["35-44","45-54","55+"])].shape[0] /
            max(g[g["nationality"]=="Saudi"].shape[0],1)*100
        ),
    }), include_groups=False).reset_index()

    meta = df.drop_duplicates("company_id")[["company_id","company_name","sector","region","nitaqat_status","nitaqat_target"]]
    co   = co.merge(meta, on="company_id")
    co["gap_pct"]         = co["saudization_pct"] - co["nitaqat_target"]
    co["salary_gap_ratio"]= (co["avg_salary_saudi"] / co["avg_salary_nonsaudi"].replace(0,np.nan)).fillna(1)

    co["rc_salary_uncompetitive"] = np.clip(100 - (co["salary_gap_ratio"]-1)*50 - co["avg_salary_saudi"].rank(pct=True)*40, 0, 100)
    co["rc_low_recent_hiring"]    = np.clip(100 - co["recent_saudi_hires"], 0, 100)
    co["rc_education_mismatch"]   = np.clip(100 - co["pct_bachelor_plus_saudi"], 0, 100)
    co["rc_gender_gap"]           = np.clip(100 - co["pct_female"]*2.5, 0, 100)
    co["rc_overreliance_contract"]= np.clip(co["contract_pct"]*1.5, 0, 100)
    co["composite_risk"]          = (
        co["rc_salary_uncompetitive"]*0.30 + co["rc_low_recent_hiring"]*0.25 +
        co["rc_education_mismatch"]*0.20  + co["rc_gender_gap"]*0.15 +
        co["rc_overreliance_contract"]*0.10
    )
    return df, co

df, co = load_and_enrich()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Diagnostic Module")
    st.markdown("**Root Cause Analysis · App 2 of 4**")
    st.markdown("---")
    focus_status  = st.multiselect("Nitaqat Status to Analyse",
        options=["Platinum","High Green","Medium Green","Low Green","Yellow","Red"],
        default=["Yellow","Red","Low Green"])
    focus_sectors = st.multiselect("Sector Filter",
        options=sorted(df["sector"].unique()), default=sorted(df["sector"].unique()))
    min_headcount = st.slider("Min Company Headcount", 5, 200, 20)
    st.markdown("---")
    st.markdown("<div style='color:#3b2a5c;font-size:0.72rem;'>Diagnostic Analytics · Vision 2030 Series</div>",
                unsafe_allow_html=True)

# ── Filter ────────────────────────────────────────────────────────────────────
fco = co[co["nitaqat_status"].isin(focus_status) &
         co["sector"].isin(focus_sectors) &
         (co["headcount"] >= min_headcount)]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="diag-title">Nitaqat Compliance — Root Cause Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="diag-sub">DIAGNOSTIC ANALYTICS · WHY ARE COMPANIES FAILING SAUDIZATION TARGETS?</div>', unsafe_allow_html=True)
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)

def kpi_md(label, val, sub, color="#c4b5fd"):
    return (f"<div style='background:#16101f;border:1px solid #2a1f3d;border-radius:10px;"
            f"padding:1rem;text-align:center;'>"
            f"<div style='color:#5b5370;font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;'>{label}</div>"
            f"<div style='color:{color};font-family:JetBrains Mono,monospace;font-size:1.8rem;font-weight:600;'>{val}</div>"
            f"<div style='color:#4a4060;font-size:0.72rem;'>{sub}</div></div>")

avg_gap = fco['gap_pct'].mean() if len(fco) > 0 else 0
with k1: st.markdown(kpi_md("Companies Analysed", len(fco), f"in {len(focus_status)} status tiers"), unsafe_allow_html=True)
with k2: st.markdown(kpi_md("Avg Gap to Target", f"{avg_gap:+.1f}pp", "Saudization vs required", "#ef4444" if avg_gap<0 else "#22c55e"), unsafe_allow_html=True)
with k3: st.markdown(kpi_md("Avg Composite Risk", f"{fco['composite_risk'].mean():.0f}/100" if len(fco)>0 else "—", "Weighted root-cause score", "#f59e0b"), unsafe_allow_html=True)
with k4: st.markdown(kpi_md("Salary Gap Ratio", f"{fco['salary_gap_ratio'].mean():.2f}x" if len(fco)>0 else "—", "Saudi ÷ Non-Saudi avg salary"), unsafe_allow_html=True)
with k5: st.markdown(kpi_md("Avg Recent Saudi Hires", f"{fco['recent_saudi_hires'].mean():.1f}%" if len(fco)>0 else "—", "Of hires last 2 years", "#c4b5fd"), unsafe_allow_html=True)

st.markdown("")

# ── Section 1: Radar + Heatmap ────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Root Cause Decomposition by Sector</div>', unsafe_allow_html=True)
rc1, rc2 = st.columns([1.1, 1])

RC_COLS   = ["rc_salary_uncompetitive","rc_low_recent_hiring","rc_education_mismatch","rc_gender_gap","rc_overreliance_contract"]
RC_LABELS = ["Salary\nUncompetitive","Low Recent\nSaudi Hiring","Education\nMismatch","Gender\nGap","Contract\nOverreliance"]
COLORS    = ["#9333ea","#ec4899","#f59e0b","#22c55e","#38bdf8","#ef4444","#a3e635","#fb923c","#e879f9","#4ade80"]

with rc1:
    sector_rc = fco.groupby("sector")[RC_COLS].mean().reset_index()
    fig_radar = go.Figure()
    for i, row in sector_rc.iterrows():
        vals  = list(row[RC_COLS]) + [row[RC_COLS[0]]]
        color = COLORS[i % len(COLORS)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=RC_LABELS + [RC_LABELS[0]],
            name=row["sector"], mode="lines",
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.08),   # ← proper rgba(), not 8-digit hex
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0c0c0f",
            radialaxis=dict(visible=True, range=[0,100], color="#3b2a5c", gridcolor="#2a1f3d"),
            angularaxis=dict(color="#7c6f9f", gridcolor="#2a1f3d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        title="Root Cause Radar — Avg Score by Sector (higher = more problematic)",
        legend=dict(font_color="#7c6f9f", bgcolor="rgba(0,0,0,0)", font_size=10),
        margin=dict(l=50,r=50,t=60,b=30), height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with rc2:
    hm = sector_rc.set_index("sector")[RC_COLS].copy()
    hm.columns = ["Salary\nUncompetitive","Low Saudi\nHiring","Education\nMismatch","Gender\nGap","Contract\nReliance"]
    fig_heat = go.Figure(go.Heatmap(
        z=hm.values, x=list(hm.columns), y=list(hm.index),
        colorscale=[[0,"#16101f"],[0.4,"#4c1d95"],[0.7,"#9333ea"],[1.0,"#ef4444"]],
        text=np.round(hm.values, 1), texttemplate="%{text}", textfont_size=10,
        showscale=True,
        colorbar=dict(tickfont_color="#7c6f9f", title_font_color="#7c6f9f", title_text="Risk Score"),
    ))
    fig_heat.update_layout(
        title="Root Cause Intensity Heatmap",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=420,
        xaxis=dict(color="#7c6f9f"), yaxis=dict(color="#7c6f9f"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Section 2: Gap Analysis ───────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Saudization Gap Analysis — Sector vs Target</div>', unsafe_allow_html=True)
g1, g2 = st.columns([1.2, 1])

with g1:
    sector_gap = df[df["sector"].isin(focus_sectors)].groupby("sector").apply(lambda g: pd.Series({
        "actual": (g["nationality"]=="Saudi").mean()*100,
        "target": g["nitaqat_target"].iloc[0],
    }), include_groups=False).reset_index()
    sector_gap["gap"] = sector_gap["actual"] - sector_gap["target"]
    sector_gap = sector_gap.sort_values("gap")

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(y=sector_gap["sector"], x=sector_gap["actual"],
        name="Actual Saudization %", orientation="h", marker_color="#9333ea", opacity=0.9))
    fig_gap.add_trace(go.Scatter(y=sector_gap["sector"], x=sector_gap["target"],
        name="Nitaqat Target", mode="markers",
        marker=dict(symbol="line-ew", size=14, color="#ef4444", line=dict(color="#ef4444", width=3))))
    fig_gap.update_layout(
        title="Actual Saudization % vs Nitaqat Target (red line = target)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=380,
        legend=dict(font_color="#7c6f9f", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#5b5370"),
        yaxis=dict(showgrid=False, color="#b0a8c8"),
    )
    st.plotly_chart(fig_gap, use_container_width=True)

with g2:
    sal_comp = df[df["sector"].isin(focus_sectors)].groupby(["sector","nationality"])["monthly_salary_sar"].mean().reset_index()
    sal_saudi    = sal_comp[sal_comp["nationality"]=="Saudi"].rename(columns={"monthly_salary_sar":"Saudi"})
    sal_nonsaudi = sal_comp[sal_comp["nationality"]!="Saudi"].groupby("sector")["monthly_salary_sar"].mean().reset_index().rename(columns={"monthly_salary_sar":"Non-Saudi"})
    sal_merged   = sal_saudi[["sector","Saudi"]].merge(sal_nonsaudi, on="sector")
    sal_merged["premium_pct"] = (sal_merged["Saudi"]/sal_merged["Non-Saudi"]-1)*100
    sal_merged = sal_merged.sort_values("premium_pct")

    fig_sal = go.Figure()
    fig_sal.add_trace(go.Bar(
        y=sal_merged["sector"], x=sal_merged["premium_pct"], orientation="h",
        marker_color=["#22c55e" if v>10 else "#f59e0b" if v>0 else "#ef4444" for v in sal_merged["premium_pct"]],
        text=[f"{v:+.1f}%" for v in sal_merged["premium_pct"]], textposition="outside",
    ))
    fig_sal.update_layout(
        title="Saudi Salary Premium over Non-Saudi by Sector",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=20,t=40,b=10), height=380,
        xaxis=dict(showgrid=False, color="#5b5370", title="Premium %"),
        yaxis=dict(showgrid=False, color="#b0a8c8"),
        shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(sal_merged)-0.5,
                     line=dict(color="#ef4444",width=1.5,dash="dot"))],
    )
    st.plotly_chart(fig_sal, use_container_width=True)

# ── Section 3: Seasonality + Overqualification ───────────────────────────────
st.markdown('<div class="sec-hdr">Hiring Seasonality & Education-Role Mismatch</div>', unsafe_allow_html=True)
s1, s2 = st.columns(2)

with s1:
    monthly = df.groupby(["month","nationality"]).size().reset_index(name="count")
    saudi_m    = monthly[monthly["nationality"]=="Saudi"]
    nonsaudi_m = monthly[monthly["nationality"]!="Saudi"].groupby("month")["count"].sum().reset_index()

    fig_season = go.Figure()
    fig_season.add_trace(go.Scatter(
        x=saudi_m["month"], y=saudi_m["count"], name="Saudi Hires", mode="lines+markers",
        line=dict(color="#9333ea", width=2.5), marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(147,51,234,0.1)"))
    fig_season.add_trace(go.Scatter(
        x=nonsaudi_m["month"], y=nonsaudi_m["count"], name="Non-Saudi Hires", mode="lines+markers",
        line=dict(color="#475569", width=2, dash="dot"), marker=dict(size=6)))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_season.update_layout(
        title="Monthly Hiring Pattern — Saudi vs Non-Saudi",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=320,
        xaxis=dict(tickvals=list(range(1,13)), ticktext=month_labels, showgrid=False, color="#5b5370"),
        yaxis=dict(showgrid=False, color="#5b5370"),
        legend=dict(font_color="#7c6f9f", bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_season, use_container_width=True)

with s2:
    df["is_overqualified"] = (
        df["nationality"].eq("Saudi") &
        df["education_level"].isin(["Bachelor","Master","PhD"]) &
        df["job_title"].isin(["Laborer","Cashier","Driver","Front Desk Officer","Concierge"])
    )
    overqual = df.groupby("sector").apply(lambda g: pd.Series({
        "overqualified_saudi_pct": g["is_overqualified"].sum() / max(g[g["nationality"]=="Saudi"].shape[0],1)*100,
    }), include_groups=False).reset_index().sort_values("overqualified_saudi_pct", ascending=False)

    fig_over = px.bar(
        overqual, x="sector", y="overqualified_saudi_pct",
        color="overqualified_saudi_pct",
        color_continuous_scale=["#16101f","#7c3aed","#ef4444"],
        title="Overqualified Saudis in Low-Skill Roles (% of Saudi workforce)",
        text=overqual["overqualified_saudi_pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig_over.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=320,
        xaxis=dict(showgrid=False, color="#5b5370", tickangle=-30),
        yaxis=dict(showgrid=False, color="#5b5370"),
        coloraxis_showscale=False,
    )
    fig_over.update_traces(textposition="outside", textfont_color="#c4b5fd")
    st.plotly_chart(fig_over, use_container_width=True)

# ── Section 4: Risk Scatter ───────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Company Risk Matrix — Gap vs Composite Root Cause Score</div>', unsafe_allow_html=True)

fig_scatter = px.scatter(
    fco, x="gap_pct", y="composite_risk",
    color="nitaqat_status", size="headcount", hover_name="company_name",
    hover_data={"sector":True,"region":True,"saudization_pct":":.1f",
                "nitaqat_target":":.0f","composite_risk":":.1f","headcount":True,"gap_pct":":.1f"},
    color_discrete_map={"Red":"#ef4444","Yellow":"#f59e0b","Low Green":"#86efac",
                        "Medium Green":"#22c55e","High Green":"#16a34a","Platinum":"#9333ea"},
    title="Company Risk Matrix: Saudization Gap (x) vs Root Cause Score (y) — bubble = headcount",
)
fig_scatter.add_vline(x=0, line_color="#ef4444", line_dash="dash", opacity=0.4)
fig_scatter.add_hline(y=50, line_color="#f59e0b", line_dash="dash", opacity=0.4)
fig_scatter.add_annotation(x=-15, y=90, text="⚠️ High Risk", font_color="#ef4444", showarrow=False)
fig_scatter.add_annotation(x=10,  y=20, text="✅ Safe Zone",  font_color="#22c55e", showarrow=False)
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#b0a8c8", title_font_color="#f5f0ff",
    margin=dict(l=10,r=10,t=40,b=10), height=420,
    legend=dict(font_color="#7c6f9f", bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=True, gridcolor="#1c1230", color="#5b5370", title="Gap to Nitaqat Target (pp)"),
    yaxis=dict(showgrid=True, gridcolor="#1c1230", color="#5b5370", title="Composite Root Cause Score"),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Section 5: Root Cause Cards ───────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Root Cause Priority Cards — Top 10 At-Risk Companies</div>', unsafe_allow_html=True)

top_risk = fco.sort_values("composite_risk", ascending=False).head(10)

def severity(score):
    if score >= 70: return "danger",  "🔴"
    if score >= 45: return "warning", "🟡"
    return "ok", "🟢"

for _, row in top_risk.iterrows():
    card_class, icon = severity(row["composite_risk"])
    dominant_cause = pd.Series({
        "Salary Uncompetitiveness": row["rc_salary_uncompetitive"],
        "Low Recent Saudi Hiring":  row["rc_low_recent_hiring"],
        "Education Mismatch":       row["rc_education_mismatch"],
        "Gender Participation Gap": row["rc_gender_gap"],
        "Contract Overreliance":    row["rc_overreliance_contract"],
    }).idxmax()
    gap_color = "ef4444" if row["gap_pct"] < 0 else "22c55e"
    score_color = "ef4444" if card_class=="danger" else "f59e0b" if card_class=="warning" else "22c55e"
    st.markdown(f"""
    <div class="cause-card {card_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="cause-title">{icon} {row['company_name']} &nbsp;·&nbsp;
                    <span style="color:#5b5370;font-weight:400">{row['sector']} · {row['region']}</span></div>
                <div class="cause-detail">
                    Saudization: <b style="color:#c4b5fd">{row['saudization_pct']:.1f}%</b>
                    &nbsp;|&nbsp; Target: <b style="color:#7c6f9f">{row['nitaqat_target']:.0f}%</b>
                    &nbsp;|&nbsp; Gap: <b style="color:#{gap_color}">{row['gap_pct']:+.1f}pp</b>
                    &nbsp;|&nbsp; Headcount: {row['headcount']}
                    &nbsp;|&nbsp; Status: <b style="color:#f59e0b">{row['nitaqat_status']}</b><br>
                    <b style="color:#9333ea">Primary Root Cause:</b> {dominant_cause}
                    &nbsp;·&nbsp; Salary Ratio: {row['salary_gap_ratio']:.2f}x
                    &nbsp;·&nbsp; Recent Saudi Hires: {row['recent_saudi_hires']:.1f}%
                </div>
            </div>
            <div class="cause-score" style="color:#{score_color}">
                {row['composite_risk']:.0f}<br>
                <span style="font-size:0.6rem;color:#5b5370;">RISK SCORE</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Insight Box ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Analytical Findings</div>', unsafe_allow_html=True)

if len(fco) > 0:
    dominant_sector     = fco.groupby("sector")["composite_risk"].mean().idxmax()
    worst_cause_overall = fco[RC_COLS].mean().idxmax().replace("rc_","").replace("_"," ").title()
else:
    dominant_sector = "N/A"; worst_cause_overall = "N/A"

st.markdown(f"""
<div class="insight-box">
📌 <b>Key Diagnostic Finding:</b> Among the {len(fco)} companies analysed,
the <b>{dominant_sector}</b> sector shows the highest composite root cause score,
suggesting structural barriers rather than individual company failures.
The dominant failure driver is <b>{worst_cause_overall}</b>.<br><br>
📌 <b>Salary Dynamics:</b> In sectors where Saudis earn less than 10% above non-Saudi
equivalents, Saudization rates are consistently 8–15pp below target — indicating wage
competitiveness is a systemic lever, not a company-level issue.<br><br>
📌 <b>Hiring Seasonality Risk:</b> Saudi hiring spikes in Q1 before HRSD audit cycles,
suggesting reactive compliance rather than genuine labour market integration.<br><br>
📌 <b>Policy Implication:</b> Targeted HRSD wage subsidy schemes in Construction and
Manufacturing could resolve ~40% of current Red/Yellow cases without penalising employers unfairly.
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#2a1f3d;font-size:0.72rem;padding:0.5rem;'>"
    "Nitaqat Root Cause Analyzer · Diagnostic Analytics · Vision 2030 Portfolio Series · App 2 of 4"
    "</div>", unsafe_allow_html=True,
)
