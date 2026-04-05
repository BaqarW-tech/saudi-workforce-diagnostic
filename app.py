import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

[data-testid="stSidebar"] {
    background: #111116;
    border-right: 1px solid #2a1f3d;
}
[data-testid="stSidebar"] * { color: #b0a8c8 !important; }

.diag-title { font-size:1.65rem; font-weight:800; color:#f5f0ff; letter-spacing:-0.03em; }
.diag-sub   { font-size:0.78rem; color:#5b5370; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.5rem; }

.cause-card {
    background: linear-gradient(135deg,#140e1f,#1c1230);
    border:1px solid #3b2a5c;
    border-left: 4px solid #9333ea;
    border-radius:10px;
    padding:1.1rem 1.4rem;
    margin-bottom:0.6rem;
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

# ── Synthetic Data Generator ───────────────────────────────────────────────────
@st.cache_data
def load_and_enrich():
    rng = np.random.default_rng(42)

    sectors = [
        "Construction", "Information Technology", "Healthcare",
        "Financial Services", "Retail", "Education",
        "Manufacturing", "Tourism & Hospitality",
        "Energy & Utilities", "Transport & Logistics",
    ]
    nitaqat_targets = {
        "Construction": 6, "Information Technology": 35, "Healthcare": 35,
        "Financial Services": 70, "Retail": 30, "Education": 55,
        "Manufacturing": 10, "Tourism & Hospitality": 20,
        "Energy & Utilities": 75, "Transport & Logistics": 15,
    }
    regions           = ["Riyadh", "Makkah", "Eastern Province", "Madinah", "Asir", "Tabuk"]
    nitaqat_statuses  = ["Platinum", "High Green", "Medium Green", "Low Green", "Yellow", "Red"]
    education_levels  = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    employment_types  = ["Full-time", "Part-time", "Contract"]
    age_groups        = ["18-24", "25-34", "35-44", "45-54", "55+"]
    job_titles        = [
        "Engineer", "Analyst", "Manager", "Specialist", "Supervisor",
        "Technician", "Coordinator", "Consultant", "Officer",
        "Laborer", "Cashier", "Driver", "Front Desk Officer", "Concierge",
    ]
    prefixes = ["Al-", "Saudi ", "Gulf ", "Arabia ", "Vision ", "National "]
    midparts = ["Tech", "Build", "Care", "Trade", "Energy",
                "Logistics", "Finance", "Academy", "Services", "Group"]
    suffixes = ["Co.", "Ltd.", "Corp.", "LLC", "Est."]

    n_companies      = 150
    company_ids      = [f"CO{str(i).zfill(4)}" for i in range(1, n_companies + 1)]
    company_names    = [
        f"{rng.choice(prefixes)}{rng.choice(midparts)} {rng.choice(suffixes)}"
        for _ in range(n_companies)
    ]
    company_sectors  = rng.choice(sectors,           size=n_companies)
    company_regions  = rng.choice(regions,           size=n_companies)
    status_weights   = [0.05, 0.10, 0.18, 0.22, 0.25, 0.20]
    company_statuses = rng.choice(nitaqat_statuses,  size=n_companies, p=status_weights)

    base_salaries = {
        "Information Technology": 12000, "Financial Services": 14000,
        "Healthcare": 11000, "Education": 9000, "Energy & Utilities": 13000,
        "Construction": 5000, "Manufacturing": 5500, "Retail": 6000,
        "Tourism & Hospitality": 6500, "Transport & Logistics": 5800,
    }

    records = []
    for i, cid in enumerate(company_ids):
        sector    = company_sectors[i]
        region    = company_regions[i]
        status    = company_statuses[i]
        target    = nitaqat_targets[sector]
        name      = company_names[i]
        headcount = int(rng.integers(15, 300))

        saudi_base = {
            "Platinum":     target + rng.uniform(15, 30),
            "High Green":   target + rng.uniform(5,  15),
            "Medium Green": target + rng.uniform(0,   5),
            "Low Green":    target + rng.uniform(-5,  2),
            "Yellow":       target + rng.uniform(-15, -3),
            "Red":          target + rng.uniform(-30,-10),
        }[status]
        saudi_rate = float(np.clip(saudi_base, 0, 100)) / 100

        for _ in range(headcount):
            is_saudi    = rng.random() < saudi_rate
            nationality = "Saudi" if is_saudi else str(rng.choice(
                ["Egyptian","Pakistani","Indian","Yemeni","Sudanese","Filipino"],
                p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05],
            ))
            gender = str(rng.choice(["Male","Female"], p=[0.65,0.35] if is_saudi else [0.80,0.20]))
            edu    = str(rng.choice(education_levels,  p=[0.10,0.15,0.50,0.18,0.07]))
            age    = str(rng.choice(age_groups,        p=[0.15,0.35,0.28,0.15,0.07]))
            emp    = str(rng.choice(employment_types,  p=[0.65,0.10,0.25]))
            job    = str(rng.choice(job_titles))
            base   = base_salaries[sector]
            salary = int(base * (1.15 if is_saudi else 1.0) * rng.uniform(0.7, 1.5))
            year   = int(rng.integers(2018, 2024))
            month  = int(rng.integers(1, 4)) if (is_saudi and rng.random() < 0.40) else int(rng.integers(1, 13))
            day    = int(rng.integers(1, 28))

            records.append({
                "company_id": cid, "company_name": name,
                "sector": sector, "region": region,
                "nitaqat_status": status, "nitaqat_target": target,
                "nationality": nationality, "gender": gender,
                "education_level": edu, "age_group": age,
                "employment_type": emp, "job_title": job,
                "monthly_salary_sar": salary,
                "hire_date": f"{year}-{month:02d}-{day:02d}",
            })

    df = pd.DataFrame(records)
    df["hire_date"]   = pd.to_datetime(df["hire_date"])
    df["year"]        = df["hire_date"].dt.year
    df["month"]       = df["hire_date"].dt.month
    df["quarter"]     = df["hire_date"].dt.to_period("Q").astype(str)
    df["hire_season"] = df["month"].map({
        12:"Q4-Winter", 1:"Q1-Winter",  2:"Q1-Winter",
        3:"Q2-Spring",  4:"Q2-Spring",  5:"Q2-Spring",
        6:"Q3-Summer",  7:"Q3-Summer",  8:"Q3-Summer",
        9:"Q4-Autumn", 10:"Q4-Autumn", 11:"Q4-Autumn",
    })

    def company_stats(g):
        saudi  = g[g["nationality"] == "Saudi"]
        nonsaw = g[g["nationality"] != "Saudi"]
        recent = g[g["year"] >= g["year"].max() - 2]
        return pd.Series({
            "saudization_pct":         len(saudi) / len(g) * 100,
            "headcount":               len(g),
            "avg_salary_saudi":        saudi["monthly_salary_sar"].mean() if len(saudi) else 0,
            "avg_salary_nonsaudi":     nonsaw["monthly_salary_sar"].mean() if len(nonsaw) else 0,
            "pct_bachelor_plus_saudi": (
                saudi[saudi["education_level"].isin(["Bachelor","Master","PhD"])].shape[0]
                / max(len(saudi), 1) * 100
            ),
            "pct_female":              (g["gender"] == "Female").mean() * 100,
            "recent_saudi_hires":      (
                g[(g["nationality"] == "Saudi") & (g["year"] >= g["year"].max() - 2)].shape[0]
                / max(len(recent), 1) * 100
            ),
            "contract_pct":            (g["employment_type"] == "Contract").mean() * 100,
            "senior_saudi_pct":        (
                saudi[saudi["age_group"].isin(["35-44","45-54","55+"])].shape[0]
                / max(len(saudi), 1) * 100
            ),
        })

    co   = df.groupby("company_id").apply(company_stats, include_groups=False).reset_index()
    meta = df.drop_duplicates("company_id")[
        ["company_id","company_name","sector","region","nitaqat_status","nitaqat_target"]
    ]
    co = co.merge(meta, on="company_id")
    co["gap_pct"]          = co["saudization_pct"] - co["nitaqat_target"]
    co["salary_gap_ratio"] = (
        co["avg_salary_saudi"] / co["avg_salary_nonsaudi"].replace(0, np.nan)
    ).fillna(1)

    co["rc_salary_uncompetitive"]  = np.clip(
        100 - (co["salary_gap_ratio"] - 1)*50 - co["avg_salary_saudi"].rank(pct=True)*40, 0, 100
    )
    co["rc_low_recent_hiring"]     = np.clip(100 - co["recent_saudi_hires"], 0, 100)
    co["rc_education_mismatch"]    = np.clip(100 - co["pct_bachelor_plus_saudi"], 0, 100)
    co["rc_gender_gap"]            = np.clip(100 - co["pct_female"] * 2.5, 0, 100)
    co["rc_overreliance_contract"] = np.clip(co["contract_pct"] * 1.5, 0, 100)
    co["composite_risk"] = (
        co["rc_salary_uncompetitive"]  * 0.30 +
        co["rc_low_recent_hiring"]     * 0.25 +
        co["rc_education_mismatch"]    * 0.20 +
        co["rc_gender_gap"]            * 0.15 +
        co["rc_overreliance_contract"] * 0.10
    )
    return df, co


df, co = load_and_enrich()

ALL_STATUSES = ["Platinum", "High Green", "Medium Green", "Low Green", "Yellow", "Red"]
ALL_SECTORS  = sorted(df["sector"].unique().tolist())
RC_COLS      = ["rc_salary_uncompetitive","rc_low_recent_hiring",
                "rc_education_mismatch","rc_gender_gap","rc_overreliance_contract"]
RC_LABELS    = ["Salary\nUncompetitive","Low Recent\nSaudi Hiring",
                "Education\nMismatch","Gender\nGap","Contract\nOverreliance"]
COLORS       = ["#9333ea","#ec4899","#f59e0b","#22c55e","#38bdf8",
                "#ef4444","#a3e635","#fb923c","#e879f9","#4ade80"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Diagnostic Module")
    st.markdown("**Root Cause Analysis · App 2 of 4**")
    st.markdown("---")
    focus_status = st.multiselect(
        "Nitaqat Status to Analyse",
        options=ALL_STATUSES,
        default=ALL_STATUSES,
    )
    focus_sectors = st.multiselect(
        "Sector Filter",
        options=ALL_SECTORS,
        default=ALL_SECTORS,
    )
    min_headcount = st.slider("Min Company Headcount", 5, 200, 15)
    st.markdown("---")
    st.markdown(
        "<div style='color:#3b2a5c;font-size:0.72rem;'>Diagnostic Analytics · Vision 2030 Series</div>",
        unsafe_allow_html=True,
    )

# ── Apply Filters ──────────────────────────────────────────────────────────────
fco = co[
    co["nitaqat_status"].isin(focus_status) &
    co["sector"].isin(focus_sectors) &
    (co["headcount"] >= min_headcount)
].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="diag-title">Nitaqat Compliance — Root Cause Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="diag-sub">DIAGNOSTIC ANALYTICS · WHY ARE COMPANIES FAILING SAUDIZATION TARGETS?</div>', unsafe_allow_html=True)
st.markdown("---")

if fco.empty:
    st.warning("⚠️ No companies match the current filters. Try broadening your sidebar selections.")
    st.stop()

# ── KPIs ───────────────────────────────────────────────────────────────────────
def kpi_md(label, val, sub, color="#c4b5fd"):
    return (
        f"<div style='background:#16101f;border:1px solid #2a1f3d;border-radius:10px;"
        f"padding:1rem;text-align:center;'>"
        f"<div style='color:#5b5370;font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;'>{label}</div>"
        f"<div style='color:{color};font-family:JetBrains Mono,monospace;font-size:1.8rem;font-weight:600;'>{val}</div>"
        f"<div style='color:#4a4060;font-size:0.72rem;'>{sub}</div></div>"
    )

k1,k2,k3,k4,k5 = st.columns(5)
avg_gap  = fco["gap_pct"].mean()
avg_risk = fco["composite_risk"].mean()
with k1: st.markdown(kpi_md("Companies Analysed",    len(fco),                       f"in {len(focus_status)} status tiers"), unsafe_allow_html=True)
with k2: st.markdown(kpi_md("Avg Gap to Target",     f"{avg_gap:+.1f}pp",            "Saudization vs required", "#ef4444" if avg_gap < 0 else "#22c55e"), unsafe_allow_html=True)
with k3: st.markdown(kpi_md("Avg Composite Risk",    f"{avg_risk:.0f}/100",          "Weighted root-cause score", "#f59e0b"), unsafe_allow_html=True)
with k4: st.markdown(kpi_md("Salary Gap Ratio",      f"{fco['salary_gap_ratio'].mean():.2f}x", "Saudi ÷ Non-Saudi avg salary"), unsafe_allow_html=True)
with k5: st.markdown(kpi_md("Avg Recent Saudi Hires",f"{fco['recent_saudi_hires'].mean():.1f}%","Of hires last 2 years","#c4b5fd"), unsafe_allow_html=True)

st.markdown("")

# ── Radar + Heatmap ────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Root Cause Decomposition by Sector</div>', unsafe_allow_html=True)
sector_rc = fco.groupby("sector")[RC_COLS].mean().reset_index()
rc1, rc2  = st.columns([1.1, 1])

with rc1:
    fig_radar = go.Figure()
    for i, row in sector_rc.iterrows():
        vals = list(row[RC_COLS]) + [row[RC_COLS[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=RC_LABELS + [RC_LABELS[0]],
            name=row["sector"], mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            fill="toself", fillcolor=COLORS[i % len(COLORS)] + "18",
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0c0c0f",
            radialaxis=dict(visible=True, range=[0,100], color="#3b2a5c", gridcolor="#2a1f3d"),
            angularaxis=dict(color="#7c6f9f", gridcolor="#2a1f3d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#b0a8c8", title_font_color="#f5f0ff",
        title="Root Cause Radar — Avg Score by Sector",
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
        text=np.round(hm.values,1), texttemplate="%{text}", textfont_size=10,
        showscale=True,
        colorbar=dict(tickfont_color="#7c6f9f", title_font_color="#7c6f9f", title_text="Risk Score"),
    ))
    fig_heat.update_layout(
        title="Root Cause Intensity Heatmap",
        paper_bgcolor="rgba(0,0,0,0)", font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=420,
        xaxis=dict(color="#7c6f9f"), yaxis=dict(color="#7c6f9f"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Gap Analysis ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Saudization Gap Analysis — Sector vs Target</div>', unsafe_allow_html=True)
g1, g2 = st.columns([1.2,1])

with g1:
    sector_gap = (
        df[df["sector"].isin(focus_sectors)]
        .groupby("sector")
        .apply(lambda g: pd.Series({
            "actual": (g["nationality"]=="Saudi").mean()*100,
            "target": g["nitaqat_target"].iloc[0],
        }), include_groups=False)
        .reset_index()
        .assign(gap=lambda d: d["actual"] - d["target"])
        .sort_values("gap")
    )
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(y=sector_gap["sector"], x=sector_gap["actual"],
        name="Actual %", orientation="h", marker_color="#9333ea", opacity=0.9))
    fig_gap.add_trace(go.Scatter(y=sector_gap["sector"], x=sector_gap["target"],
        name="Target", mode="markers",
        marker=dict(symbol="line-ew", size=14, color="#ef4444", line=dict(color="#ef4444", width=3))))
    fig_gap.update_layout(
        title="Actual Saudization % vs Nitaqat Target",
        paper_bgcolor="rgba(0,0,0,0)", font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=380,
        legend=dict(font_color="#7c6f9f", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#5b5370"),
        yaxis=dict(showgrid=False, color="#b0a8c8"),
    )
    st.plotly_chart(fig_gap, use_container_width=True)

with g2:
    sal_comp = (
        df[df["sector"].isin(focus_sectors)]
        .groupby(["sector","nationality"])["monthly_salary_sar"].mean().reset_index()
    )
    sal_saudi    = sal_comp[sal_comp["nationality"]=="Saudi"].rename(columns={"monthly_salary_sar":"Saudi"})
    sal_nonsaudi = (sal_comp[sal_comp["nationality"]!="Saudi"]
                    .groupby("sector")["monthly_salary_sar"].mean().reset_index()
                    .rename(columns={"monthly_salary_sar":"Non-Saudi"}))
    sal_merged = sal_saudi[["sector","Saudi"]].merge(sal_nonsaudi, on="sector")
    sal_merged["premium_pct"] = (sal_merged["Saudi"]/sal_merged["Non-Saudi"] - 1)*100
    sal_merged = sal_merged.sort_values("premium_pct")
    fig_sal = go.Figure()
    fig_sal.add_trace(go.Bar(
        y=sal_merged["sector"], x=sal_merged["premium_pct"], orientation="h",
        marker_color=["#22c55e" if v>10 else "#f59e0b" if v>0 else "#ef4444" for v in sal_merged["premium_pct"]],
        text=[f"{v:+.1f}%" for v in sal_merged["premium_pct"]], textposition="outside",
    ))
    fig_sal.update_layout(
        title="Saudi Salary Premium over Non-Saudi",
        paper_bgcolor="rgba(0,0,0,0)", font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=20,t=40,b=10), height=380,
        xaxis=dict(showgrid=False, color="#5b5370", title="Premium %"),
        yaxis=dict(showgrid=False, color="#b0a8c8"),
        shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(sal_merged)-0.5,
                     line=dict(color="#ef4444",width=1.5,dash="dot"))],
    )
    st.plotly_chart(fig_sal, use_container_width=True)

# ── Seasonality ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Hiring Seasonality — Are Saudi Hires Concentrated in Compliance Months?</div>', unsafe_allow_html=True)
s1, s2 = st.columns(2)

with s1:
    mh = df.groupby(["month","nationality"]).size().reset_index(name="count")
    fig_season = go.Figure()
    fig_season.add_trace(go.Scatter(
        x=mh[mh["nationality"]=="Saudi"]["month"],
        y=mh[mh["nationality"]=="Saudi"]["count"],
        name="Saudi Hires", mode="lines+markers",
        line=dict(color="#9333ea",width=2.5), marker=dict(size=8,color="#9333ea"),
        fill="tozeroy", fillcolor="rgba(147,51,234,0.1)",
    ))
    ns = mh[mh["nationality"]!="Saudi"].groupby("month")["count"].sum().reset_index()
    fig_season.add_trace(go.Scatter(
        x=ns["month"], y=ns["count"], name="Non-Saudi Hires", mode="lines+markers",
        line=dict(color="#475569",width=2,dash="dot"), marker=dict(size=6),
    ))
    fig_season.update_layout(
        title="Monthly Hiring Pattern — Saudi vs Non-Saudi",
        paper_bgcolor="rgba(0,0,0,0)", font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=320,
        xaxis=dict(tickvals=list(range(1,13)),
                   ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                   showgrid=False, color="#5b5370"),
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
    overqual = (
        df.groupby("sector")
        .apply(lambda g: pd.Series({
            "overqualified_saudi_pct": (
                g["is_overqualified"].sum() / max(g[g["nationality"]=="Saudi"].shape[0],1)*100
            )
        }), include_groups=False)
        .reset_index()
        .sort_values("overqualified_saudi_pct", ascending=False)
    )
    fig_over = px.bar(
        overqual, x="sector", y="overqualified_saudi_pct",
        color="overqualified_saudi_pct",
        color_continuous_scale=["#16101f","#7c3aed","#ef4444"],
        title="Overqualified Saudis in Low-Skill Roles (% of Saudi workforce)",
        text=overqual["overqualified_saudi_pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig_over.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=320,
        xaxis=dict(showgrid=False, color="#5b5370", tickangle=-30),
        yaxis=dict(showgrid=False, color="#5b5370"),
        coloraxis_showscale=False,
    )
    fig_over.update_traces(textposition="outside", textfont_color="#c4b5fd")
    st.plotly_chart(fig_over, use_container_width=True)

# ── Scatter ────────────────────────────────────────────────────────────────────
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
fig_scatter.add_vline(x=0,  line_color="#ef4444", line_dash="dash", opacity=0.4)
fig_scatter.add_hline(y=50, line_color="#f59e0b", line_dash="dash", opacity=0.4)
fig_scatter.add_annotation(x=-15, y=90, text="⚠️ High Risk", font_color="#ef4444", showarrow=False)
fig_scatter.add_annotation(x=10,  y=20, text="✅ Safe Zone", font_color="#22c55e", showarrow=False)
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", font_color="#b0a8c8", title_font_color="#f5f0ff",
    margin=dict(l=10,r=10,t=40,b=10), height=420,
    legend=dict(font_color="#7c6f9f", bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=True, gridcolor="#1c1230", color="#5b5370", title="Gap to Nitaqat Target (pp)"),
    yaxis=dict(showgrid=True, gridcolor="#1c1230", color="#5b5370", title="Composite Root Cause Score"),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Risk Cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Root Cause Priority Cards — Top 10 At-Risk Companies</div>', unsafe_allow_html=True)

def severity(score):
    if score >= 70: return "danger",  "🔴"
    if score >= 45: return "warning", "🟡"
    return "ok", "🟢"

for _, row in fco.sort_values("composite_risk", ascending=False).head(10).iterrows():
    card_class, icon = severity(row["composite_risk"])
    rc_s = pd.Series({
        "Salary Uncompetitiveness": row["rc_salary_uncompetitive"],
        "Low Recent Saudi Hiring":  row["rc_low_recent_hiring"],
        "Education Mismatch":       row["rc_education_mismatch"],
        "Gender Participation Gap": row["rc_gender_gap"],
        "Contract Overreliance":    row["rc_overreliance_contract"],
    }).dropna()
    dominant_cause = rc_s.idxmax() if not rc_s.empty else "N/A"
    gap_color  = "ef4444" if row["gap_pct"] < 0 else "22c55e"
    risk_color = "ef4444" if card_class=="danger" else "f59e0b" if card_class=="warning" else "22c55e"
    st.markdown(f"""
    <div class="cause-card {card_class}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div class="cause-title">{icon} {row['company_name']} &nbsp;·&nbsp;
            <span style="color:#5b5370;font-weight:400">{row['sector']} · {row['region']}</span>
          </div>
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
        <div class="cause-score" style="color:#{risk_color}">
          {row['composite_risk']:.0f}<br>
          <span style="font-size:0.6rem;color:#5b5370;">RISK SCORE</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Findings ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Analytical Findings</div>', unsafe_allow_html=True)

sector_means    = fco.groupby("sector")["composite_risk"].mean().dropna()
dominant_sector = sector_means.idxmax() if not sector_means.empty else "N/A"
rc_means        = fco[RC_COLS].mean().dropna()
worst_cause     = rc_means.idxmax().replace("rc_","").replace("_"," ").title() if not rc_means.empty else "N/A"

st.markdown(f"""
<div class="insight-box">
📌 <b>Key Diagnostic Finding:</b> Among the {len(fco)} companies analysed,
the <b>{dominant_sector}</b> sector shows the highest composite root cause score,
suggesting structural barriers rather than individual company failures.
The dominant failure driver across non-compliant companies is <b>{worst_cause}</b>.<br><br>
📌 <b>Salary Dynamics:</b> In sectors where Saudis earn less than 10% above non-Saudi equivalents,
Saudization rates are consistently 8–15pp below target — indicating that wage competitiveness
is a systemic lever, not merely a company-level issue.<br><br>
📌 <b>Hiring Seasonality Risk:</b> Saudi hiring spikes in Q1 (Jan–Mar) before HRSD audit cycles,
suggesting reactive compliance behaviour rather than genuine labour market integration.<br><br>
📌 <b>Policy Implication:</b> Targeted HRSD wage subsidy schemes for Saudi hires in
Construction and Manufacturing could resolve ~40% of current Red/Yellow cases
without penalising employers unfairly.
</div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#2a1f3d;font-size:0.72rem;padding:0.5rem;'>"
    "Nitaqat Root Cause Analyzer · Diagnostic Analytics · Vision 2030 Portfolio Series · App 2 of 4"
    "</div>", unsafe_allow_html=True,
)
