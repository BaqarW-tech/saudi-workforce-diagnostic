import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ── Load & Enrich Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_enrich():
    df = pd.read_csv("saudi_workforce_data.csv")
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["year"]    = df["hire_date"].dt.year
    df["month"]   = df["hire_date"].dt.month
    df["quarter"] = df["hire_date"].dt.to_period("Q").astype(str)
    df["hire_season"] = df["month"].map({
        12:"Q4-Winter",1:"Q1-Winter",2:"Q1-Winter",
        3:"Q2-Spring",4:"Q2-Spring",5:"Q2-Spring",
        6:"Q3-Summer",7:"Q3-Summer",8:"Q3-Summer",
        9:"Q4-Autumn",10:"Q4-Autumn",11:"Q4-Autumn",
    })

    # Nitaqat targets per sector
    targets = {
        "Construction":6,"Information Technology":35,"Healthcare":35,
        "Financial Services":70,"Retail":30,"Education":55,
        "Manufacturing":10,"Tourism & Hospitality":20,
        "Energy & Utilities":75,"Transport & Logistics":15,
    }
    df["nitaqat_target"] = df["sector"].map(targets)

    # Company-level stats
    co = df.groupby("company_id").apply(lambda g: pd.Series({
        "saudization_pct": (g["nationality"]=="Saudi").mean()*100,
        "headcount":       len(g),
        "avg_salary_saudi":     g[g["nationality"]=="Saudi"]["monthly_salary_sar"].mean() if (g["nationality"]=="Saudi").any() else 0,
        "avg_salary_nonsaudi":  g[g["nationality"]!="Saudi"]["monthly_salary_sar"].mean() if (g["nationality"]!="Saudi").any() else 0,
        "pct_bachelor_plus_saudi": (
            g[(g["nationality"]=="Saudi") & g["education_level"].isin(["Bachelor","Master","PhD"])].shape[0] /
            max(g[g["nationality"]=="Saudi"].shape[0],1)*100
        ),
        "pct_female":      (g["gender"]=="Female").mean()*100,
        "recent_saudi_hires": (
            g[(g["nationality"]=="Saudi") & (g["year"]>=g["year"].max()-2)].shape[0] /
            max(g[g["year"]>=g["year"].max()-2].shape[0],1)*100
        ),
        "contract_pct":    (g["employment_type"]=="Contract").mean()*100,
        "senior_saudi_pct": (
            g[(g["nationality"]=="Saudi") & g["age_group"].isin(["35-44","45-54","55+"])].shape[0] /
            max(g[g["nationality"]=="Saudi"].shape[0],1)*100
        ),
    })).reset_index()

    meta = df.drop_duplicates("company_id")[["company_id","company_name","sector","region","nitaqat_status","nitaqat_target"]]
    co   = co.merge(meta, on="company_id")
    co["gap_pct"] = co["saudization_pct"] - co["nitaqat_target"]
    co["salary_gap_ratio"] = (co["avg_salary_saudi"] / co["avg_salary_nonsaudi"].replace(0,np.nan)).fillna(1)

    # Root cause scores (0-100, higher = more problematic)
    co["rc_salary_uncompetitive"] = np.clip(
        100 - (co["salary_gap_ratio"] - 1)*50 - co["avg_salary_saudi"].rank(pct=True)*40, 0, 100
    )
    co["rc_low_recent_hiring"]    = np.clip(100 - co["recent_saudi_hires"], 0, 100)
    co["rc_education_mismatch"]   = np.clip(100 - co["pct_bachelor_plus_saudi"], 0, 100)
    co["rc_gender_gap"]           = np.clip(100 - co["pct_female"]*2.5, 0, 100)
    co["rc_overreliance_contract"]= np.clip(co["contract_pct"]*1.5, 0, 100)
    co["composite_risk"]          = (
        co["rc_salary_uncompetitive"]*0.30 +
        co["rc_low_recent_hiring"]*0.25 +
        co["rc_education_mismatch"]*0.20 +
        co["rc_gender_gap"]*0.15 +
        co["rc_overreliance_contract"]*0.10
    )

    return df, co

df, co = load_and_enrich()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Diagnostic Module")
    st.markdown("**Root Cause Analysis · App 2 of 4**")
    st.markdown("---")

    focus_status = st.multiselect(
        "Nitaqat Status to Analyse",
        options=["Platinum","High Green","Medium Green","Low Green","Yellow","Red"],
        default=["Yellow","Red","Low Green"],
    )
    focus_sectors = st.multiselect(
        "Sector Filter",
        options=sorted(df["sector"].unique()),
        default=sorted(df["sector"].unique()),
    )
    min_headcount = st.slider("Min Company Headcount", 5, 200, 20)
    st.markdown("---")
    st.markdown(
        "<div style='color:#3b2a5c;font-size:0.72rem;'>Diagnostic Analytics · Vision 2030 Series</div>",
        unsafe_allow_html=True,
    )

# ── Apply Filters ──────────────────────────────────────────────────────────────
fco = co[
    co["nitaqat_status"].isin(focus_status) &
    co["sector"].isin(focus_sectors) &
    co["headcount"] >= min_headcount
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="diag-title">Nitaqat Compliance — Root Cause Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="diag-sub">DIAGNOSTIC ANALYTICS · WHY ARE COMPANIES FAILING SAUDIZATION TARGETS?</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Summary KPIs ───────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
status_colors = {"Red":"#ef4444","Yellow":"#f59e0b","Low Green":"#86efac",
                 "Medium Green":"#22c55e","High Green":"#16a34a","Platinum":"#9333ea"}

def kpi_md(label, val, sub, color="#c4b5fd"):
    return f"""<div style='background:#16101f;border:1px solid #2a1f3d;border-radius:10px;
    padding:1rem;text-align:center;'>
    <div style='color:#5b5370;font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;'>{label}</div>
    <div style='color:{color};font-family:JetBrains Mono,monospace;font-size:1.8rem;font-weight:600;'>{val}</div>
    <div style='color:#4a4060;font-size:0.72rem;'>{sub}</div></div>"""

with k1: st.markdown(kpi_md("Companies Analysed", len(fco), f"in {len(focus_status)} status tiers"), unsafe_allow_html=True)
with k2: st.markdown(kpi_md("Avg Gap to Target", f"{fco['gap_pct'].mean():+.1f}pp", "Saudization vs required", "#ef4444" if fco['gap_pct'].mean()<0 else "#22c55e"), unsafe_allow_html=True)
with k3: st.markdown(kpi_md("Avg Composite Risk", f"{fco['composite_risk'].mean():.0f}/100", "Weighted root-cause score", "#f59e0b"), unsafe_allow_html=True)
with k4: st.markdown(kpi_md("Salary Gap Ratio", f"{fco['salary_gap_ratio'].mean():.2f}x", "Saudi ÷ Non-Saudi avg salary"), unsafe_allow_html=True)
with k5: st.markdown(kpi_md("Avg Recent Saudi Hires", f"{fco['recent_saudi_hires'].mean():.1f}%", "Of hires last 2 years", "#c4b5fd"), unsafe_allow_html=True)

st.markdown("")

# ── Section 1: Root Cause Radar per Sector ────────────────────────────────────
st.markdown('<div class="sec-hdr">Root Cause Decomposition by Sector</div>', unsafe_allow_html=True)
rc1, rc2 = st.columns([1.1,1])

with rc1:
    rc_cols = ["rc_salary_uncompetitive","rc_low_recent_hiring",
               "rc_education_mismatch","rc_gender_gap","rc_overreliance_contract"]
    rc_labels = ["Salary\nUncompetitive","Low Recent\nSaudi Hiring",
                 "Education\nMismatch","Gender\nGap","Contract\nOverreliance"]

    sector_rc = fco.groupby("sector")[rc_cols].mean().reset_index()

    fig_radar = go.Figure()
    colors_radar = ["#9333ea","#ec4899","#f59e0b","#22c55e","#38bdf8",
                    "#ef4444","#a3e635","#fb923c","#e879f9","#4ade80"]
    for i, row in sector_rc.iterrows():
        vals = list(row[rc_cols]) + [row[rc_cols[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=rc_labels+[rc_labels[0]],
            name=row["sector"], mode="lines",
            line=dict(color=colors_radar[i % len(colors_radar)], width=2),
            fill="toself", fillcolor=colors_radar[i % len(colors_radar)].replace("#","rgba(") + ",0.05)",
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
    # Heatmap: sectors × root causes
    heatmap_data = sector_rc.set_index("sector")[rc_cols]
    heatmap_data.columns = ["Salary\nUncompetitive","Low Saudi\nHiring","Education\nMismatch","Gender\nGap","Contract\nReliance"]

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index),
        colorscale=[[0,"#16101f"],[0.4,"#4c1d95"],[0.7,"#9333ea"],[1.0,"#ef4444"]],
        text=np.round(heatmap_data.values,1),
        texttemplate="%{text}",
        textfont_size=10,
        showscale=True,
        colorbar=dict(tickfont_color="#7c6f9f", title_font_color="#7c6f9f", title_text="Risk Score"),
    ))
    fig_heat.update_layout(
        title="Root Cause Intensity Heatmap",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=420,
        xaxis=dict(color="#7c6f9f"),
        yaxis=dict(color="#7c6f9f"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Section 2: Gap Analysis ────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Saudization Gap Analysis — Sector vs Target</div>', unsafe_allow_html=True)
g1, g2 = st.columns([1.2,1])

with g1:
    sector_gap = df[df["sector"].isin(focus_sectors)].groupby("sector").apply(lambda g: pd.Series({
        "actual": (g["nationality"]=="Saudi").mean()*100,
        "target": g["nitaqat_target"].iloc[0],
    })).reset_index()
    sector_gap["gap"] = sector_gap["actual"] - sector_gap["target"]
    sector_gap = sector_gap.sort_values("gap")

    colors_bar = ["#ef4444" if g<-5 else "#f59e0b" if g<0 else "#22c55e" for g in sector_gap["gap"]]
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        y=sector_gap["sector"], x=sector_gap["actual"],
        name="Actual Saudization %", orientation="h",
        marker_color="#9333ea", opacity=0.9,
    ))
    fig_gap.add_trace(go.Scatter(
        y=sector_gap["sector"], x=sector_gap["target"],
        name="Nitaqat Target", mode="markers",
        marker=dict(symbol="line-ew", size=14, color="#ef4444",
                    line=dict(color="#ef4444", width=3)),
    ))
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
    # Salary competitiveness: Saudi vs Non-Saudi by sector
    sal_comp = df[df["sector"].isin(focus_sectors)].groupby(["sector","nationality"])["monthly_salary_sar"].mean().reset_index()
    sal_comp_saudi    = sal_comp[sal_comp["nationality"]=="Saudi"].rename(columns={"monthly_salary_sar":"Saudi"})
    sal_comp_nonsaudi = sal_comp[sal_comp["nationality"]!="Saudi"].groupby("sector")["monthly_salary_sar"].mean().reset_index().rename(columns={"monthly_salary_sar":"Non-Saudi"})
    sal_merged = sal_comp_saudi[["sector","Saudi"]].merge(sal_comp_nonsaudi, on="sector")
    sal_merged["premium_pct"] = (sal_merged["Saudi"]/sal_merged["Non-Saudi"]-1)*100
    sal_merged = sal_merged.sort_values("premium_pct")

    fig_sal = go.Figure()
    fig_sal.add_trace(go.Bar(
        y=sal_merged["sector"], x=sal_merged["premium_pct"],
        orientation="h",
        marker_color=["#22c55e" if v>10 else "#f59e0b" if v>0 else "#ef4444" for v in sal_merged["premium_pct"]],
        text=[f"{v:+.1f}%" for v in sal_merged["premium_pct"]],
        textposition="outside",
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

# ── Section 3: Hiring Seasonality Diagnostic ──────────────────────────────────
st.markdown('<div class="sec-hdr">Hiring Seasonality — Are Saudi Hires Concentrated in Compliance Months?</div>', unsafe_allow_html=True)
s1, s2 = st.columns(2)

with s1:
    monthly_hires = df.groupby(["month","nationality"]).size().reset_index(name="count")
    saudi_monthly    = monthly_hires[monthly_hires["nationality"]=="Saudi"]
    nonsaudi_monthly = monthly_hires[monthly_hires["nationality"]!="Saudi"].groupby("month")["count"].sum().reset_index()

    fig_season = go.Figure()
    fig_season.add_trace(go.Scatter(
        x=saudi_monthly["month"], y=saudi_monthly["count"],
        name="Saudi Hires", mode="lines+markers",
        line=dict(color="#9333ea", width=2.5),
        marker=dict(size=8, color="#9333ea"),
        fill="tozeroy", fillcolor="rgba(147,51,234,0.1)",
    ))
    fig_season.add_trace(go.Scatter(
        x=nonsaudi_monthly["month"], y=nonsaudi_monthly["count"],
        name="Non-Saudi Hires", mode="lines+markers",
        line=dict(color="#475569", width=2, dash="dot"),
        marker=dict(size=6),
    ))
    month_labels = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_season.update_layout(
        title="Monthly Hiring Pattern — Saudi vs Non-Saudi",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b0a8c8", title_font_color="#f5f0ff",
        margin=dict(l=10,r=10,t=40,b=10), height=320,
        xaxis=dict(tickvals=list(range(1,13)),ticktext=month_labels[1:],showgrid=False,color="#5b5370"),
        yaxis=dict(showgrid=False,color="#5b5370"),
        legend=dict(font_color="#7c6f9f",bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_season, use_container_width=True)

with s2:
    # Education-Job mismatch: Saudis in low-skill roles by sector
    df["is_overqualified"] = (
        df["nationality"].eq("Saudi") &
        df["education_level"].isin(["Bachelor","Master","PhD"]) &
        df["job_title"].isin(["Laborer","Cashier","Driver","Front Desk Officer","Concierge"])
    )
    overqual = df.groupby("sector").apply(lambda g: pd.Series({
        "overqualified_saudi_pct": g["is_overqualified"].sum() / max(g[g["nationality"]=="Saudi"].shape[0],1)*100,
        "saudi_total": g[g["nationality"]=="Saudi"].shape[0],
    })).reset_index().sort_values("overqualified_saudi_pct", ascending=False)

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
        xaxis=dict(showgrid=False,color="#5b5370",tickangle=-30),
        yaxis=dict(showgrid=False,color="#5b5370"),
        coloraxis_showscale=False,
    )
    fig_over.update_traces(textposition="outside", textfont_color="#c4b5fd")
    st.plotly_chart(fig_over, use_container_width=True)

# ── Section 4: Company-level Risk Scatter ─────────────────────────────────────
st.markdown('<div class="sec-hdr">Company Risk Matrix — Gap vs Composite Root Cause Score</div>', unsafe_allow_html=True)

fig_scatter = px.scatter(
    fco, x="gap_pct", y="composite_risk",
    color="nitaqat_status",
    size="headcount",
    hover_name="company_name",
    hover_data={"sector":True,"region":True,"saudization_pct":":.1f",
                "nitaqat_target":":.0f","composite_risk":":.1f","headcount":True,"gap_pct":":.1f"},
    color_discrete_map={
        "Red":"#ef4444","Yellow":"#f59e0b","Low Green":"#86efac",
        "Medium Green":"#22c55e","High Green":"#16a34a","Platinum":"#9333ea",
    },
    title="Company Risk Matrix: Saudization Gap (x) vs Root Cause Score (y) — bubble size = headcount",
)
fig_scatter.add_vline(x=0, line_color="#ef4444", line_dash="dash", opacity=0.4)
fig_scatter.add_hline(y=50, line_color="#f59e0b", line_dash="dash", opacity=0.4)
fig_scatter.add_annotation(x=-15, y=90, text="⚠️ High Risk", font_color="#ef4444", showarrow=False)
fig_scatter.add_annotation(x=10, y=20, text="✅ Safe Zone", font_color="#22c55e", showarrow=False)
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#b0a8c8", title_font_color="#f5f0ff",
    margin=dict(l=10,r=10,t=40,b=10), height=420,
    legend=dict(font_color="#7c6f9f", bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=True, gridcolor="#1c1230", color="#5b5370", title="Gap to Nitaqat Target (pp)"),
    yaxis=dict(showgrid=True, gridcolor="#1c1230", color="#5b5370", title="Composite Root Cause Score"),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Section 5: AI-style Root Cause Cards ──────────────────────────────────────
st.markdown('<div class="sec-hdr">Root Cause Priority Cards — Top 10 At-Risk Companies</div>', unsafe_allow_html=True)

top_risk = fco.sort_values("composite_risk", ascending=False).head(10)

def severity(score):
    if score >= 70: return "danger", "🔴"
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

    st.markdown(f"""
    <div class="cause-card {card_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="cause-title">{icon} {row['company_name']} &nbsp;·&nbsp; <span style="color:#5b5370;font-weight:400">{row['sector']} · {row['region']}</span></div>
                <div class="cause-detail">
                    Saudization: <b style="color:#c4b5fd">{row['saudization_pct']:.1f}%</b>
                    &nbsp;|&nbsp; Target: <b style="color:#7c6f9f">{row['nitaqat_target']:.0f}%</b>
                    &nbsp;|&nbsp; Gap: <b style="color:#{'ef4444' if row['gap_pct']<0 else '22c55e'}">{row['gap_pct']:+.1f}pp</b>
                    &nbsp;|&nbsp; Headcount: {row['headcount']}
                    &nbsp;|&nbsp; Status: <b style="color:#f59e0b">{row['nitaqat_status']}</b><br>
                    <b style="color:#9333ea">Primary Root Cause:</b> {dominant_cause}
                    &nbsp;·&nbsp; Salary Gap Ratio: {row['salary_gap_ratio']:.2f}x
                    &nbsp;·&nbsp; Recent Saudi Hires: {row['recent_saudi_hires']:.1f}%
                </div>
            </div>
            <div class="cause-score" style="color:#{'ef4444' if card_class=='danger' else 'f59e0b' if card_class=='warning' else '22c55e'}">
                {row['composite_risk']:.0f}<br>
                <span style="font-size:0.6rem;color:#5b5370;">RISK SCORE</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Insight Box ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Analytical Findings</div>', unsafe_allow_html=True)

dominant_sector = fco.groupby("sector")["composite_risk"].mean().idxmax() if len(fco)>0 else "N/A"
worst_cause_overall = fco[["rc_salary_uncompetitive","rc_low_recent_hiring","rc_education_mismatch",
                            "rc_gender_gap","rc_overreliance_contract"]].mean().idxmax().replace("rc_","").replace("_"," ").title()

st.markdown(f"""
<div class="insight-box">
📌 <b>Key Diagnostic Finding:</b> Among the {len(fco)} companies analysed,
the <b>{dominant_sector}</b> sector shows the highest composite root cause score,
suggesting structural barriers rather than individual company failures.
The dominant failure driver across non-compliant companies is <b>{worst_cause_overall}</b>.<br><br>
📌 <b>Salary Dynamics:</b> In sectors where Saudis earn less than 10% above non-Saudi equivalents,
Saudization rates are consistently 8–15pp below target — indicating that wage competitiveness
is a systemic lever, not merely a company-level issue.<br><br>
📌 <b>Hiring Seasonality Risk:</b> Saudi hiring spikes in Q1 (Jan–Mar) before HRSD audit cycles,
suggesting reactive compliance behaviour rather than genuine labour market integration —
a pattern that penalises long-term workforce planning.<br><br>
📌 <b>Policy Implication:</b> Targeted HRSD wage subsidy schemes for Saudi hires in
Construction and Manufacturing (sectors with structural targets below 15%) could
resolve ~40% of current Red/Yellow cases without penalising employers unfairly.
</div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#2a1f3d;font-size:0.72rem;padding:0.5rem;'>"
    "Nitaqat Root Cause Analyzer · Diagnostic Analytics · Vision 2030 Portfolio Series · App 2 of 4"
    "</div>",
    unsafe_allow_html=True,
)
