import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="BNI Customer Opportunity Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --navy: #071A33;
        --navy-2: #0D2948;
        --orange: #F28C28;
        --orange-2: #FFB45B;
        --ink: #10233F;
        --muted: #6B7A90;
        --line: #E7ECF2;
        --bg: #F5F7FA;
        --white: #FFFFFF;
        --green: #198754;
        --red: #C94C4C;
    }

    .stApp { background: var(--bg); }
    [data-testid="stHeader"] { background: rgba(245,247,250,0.92); }
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1450px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06162C 0%, #0B2340 100%);
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #EAF1F8 !important; }
    [data-testid="stSidebar"] .stRadio > label { color: #AFC0D5 !important; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: .35rem; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: .62rem .7rem; border-radius: .65rem; transition: .15s;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display:none; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
        background: rgba(242,140,40,.18); border-left: 3px solid var(--orange); 
    }

    .brand { padding: .35rem .25rem 1.4rem .25rem; }
    .brand-mark { font-size: 1.55rem; font-weight: 800; letter-spacing: -.03em; }
    .brand-mark span { color: var(--orange); }
    .brand-sub { color: #9FB2C8 !important; font-size: .72rem; margin-top: .15rem; }
    .sidebar-footer { position: fixed; bottom: 1.5rem; color: #8196AE !important; font-size: .68rem; }

    .hero {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 72%, #173C60 100%);
        border-radius: 18px; padding: 1.55rem 1.7rem; color: white;
        margin-bottom: 1.15rem; box-shadow: 0 10px 30px rgba(7,26,51,.12);
    }
    .hero-kicker { color: #FFB45B; font-size: .73rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .hero-title { font-size: 2rem; line-height: 1.1; font-weight: 800; letter-spacing: -.04em; margin-top: .28rem; }
    .hero-sub { color: #C8D6E5; font-size: .9rem; margin-top: .45rem; max-width: 850px; }

    .section { display:flex; align-items:center; justify-content:space-between; margin: 1.15rem 0 .65rem; }
    .section-title { color: var(--ink); font-size: 1.05rem; font-weight: 800; letter-spacing: -.02em; }
    .section-caption { color: var(--muted); font-size: .72rem; }

    .metric-card {
        background: var(--white); border: 1px solid var(--line); border-radius: 14px;
        padding: 1rem 1.05rem; min-height: 105px; box-shadow: 0 4px 16px rgba(16,35,63,.04);
    }
    .metric-label { color: var(--muted); font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { color: var(--ink); font-size: 1.55rem; font-weight: 800; margin-top: .25rem; letter-spacing: -.035em; }
    .metric-note { color: #8492A6; font-size: .7rem; margin-top: .2rem; }

    .panel { background: var(--white); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.05rem .75rem; box-shadow: 0 4px 16px rgba(16,35,63,.035); }
    .panel-title { color: var(--ink); font-size: .92rem; font-weight: 800; margin-bottom: .1rem; }
    .panel-sub { color: var(--muted); font-size: .7rem; margin-bottom: .6rem; }

    .rec-card { background:#fff; border:1px solid var(--line); border-left:4px solid var(--orange); border-radius:11px; padding:.72rem .8rem; margin-bottom:.55rem; }
    .rec-rank { color:var(--orange); font-size:.68rem; font-weight:800; }
    .rec-name { color:var(--ink); font-size:.88rem; font-weight:800; margin:.08rem 0 .25rem; }
    .rec-score { color:var(--muted); font-size:.7rem; }

    .tag { display:inline-block; padding:.25rem .55rem; border-radius:999px; background:#FFF2E3; color:#B7610B; font-size:.68rem; font-weight:800; }

    [data-testid="stMetric"] { background: transparent; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .stButton button { border-radius: 9px; }
    div[data-baseweb="select"] > div { border-radius: 9px; border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_data():
    files = [
        "customers.csv", "products.csv", "customer_products.csv",
        "transactions.csv", "monthly_customer_behavior.csv",
        "customer_feature_table.csv", "future_product_targets.csv",
    ]
    missing = [f for f in files if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError("Missing files in data/: " + ", ".join(missing))
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    customer_products = pd.read_csv(DATA_DIR / "customer_products.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    monthly_behavior = pd.read_csv(DATA_DIR / "monthly_customer_behavior.csv")
    customer_features = pd.read_csv(DATA_DIR / "customer_feature_table.csv")
    future_targets = pd.read_csv(DATA_DIR / "future_product_targets.csv")
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    return customers, products, customer_products, transactions, monthly_behavior, customer_features, future_targets

try:
    customers, products, customer_products, transactions, monthly_behavior, customer_features, future_targets = load_data()
except Exception as e:
    st.error(str(e))
    st.info("Pastikan 7 file CSV berada di folder data/ dalam repository.")
    st.stop()

# ============================================================
# SEGMENTATION
# ============================================================
@st.cache_data
def build_segmentation(customer_features):
    cols = [
        "annual_revenue", "avg_balance", "employees", "total_outgoing_amount",
        "total_outgoing_tx", "avg_outgoing_amount", "unique_outgoing_counterparties",
        "total_incoming_amount", "total_incoming_tx", "avg_incoming_amount",
        "unique_incoming_counterparties", "product_count",
    ]
    X = customer_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = StandardScaler().fit_transform(np.log1p(X))
    labels = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X_scaled)
    return pd.DataFrame({"customer_id": customer_features["customer_id"], "segment": labels})

segment_result = build_segmentation(customer_features)
customers = customers.merge(segment_result, on="customer_id", how="left")
segment_names = {0: "High Value Business", 1: "Transactional Business", 2: "Emerging Business", 3: "Network Connected"}
customers["segment_name"] = customers["segment"].map(segment_names)

# ============================================================
# GRAPH
# ============================================================
@st.cache_resource
def build_graph(transactions, customer_ids):
    graph = nx.DiGraph()
    graph.add_nodes_from(customer_ids)
    edges = transactions.groupby(["sender_customer_id", "receiver_customer_id"]).agg(
        transaction_count=("transaction_id", "count"), total_amount=("amount", "sum")
    ).reset_index()
    graph.add_edges_from([
        (r.sender_customer_id, r.receiver_customer_id, {
            "transaction_count": r.transaction_count, "total_amount": r.total_amount
        }) for r in edges.itertuples(index=False)
    ])
    return graph

G = build_graph(transactions, customers["customer_id"].tolist())

@st.cache_data
def build_graph_features(transactions, customer_ids):
    graph = nx.DiGraph(); graph.add_nodes_from(customer_ids)
    edges = transactions.groupby(["sender_customer_id", "receiver_customer_id"])["amount"].sum().reset_index(name="total_amount")
    graph.add_edges_from([(r.sender_customer_id, r.receiver_customer_id, {"total_amount": r.total_amount}) for r in edges.itertuples(index=False)])
    pagerank = nx.pagerank(graph, weight="total_amount")
    return pd.DataFrame({
        "customer_id": customer_ids,
        "in_degree": [graph.in_degree(c) for c in customer_ids],
        "out_degree": [graph.out_degree(c) for c in customer_ids],
        "pagerank": [pagerank.get(c, 0) for c in customer_ids],
    })

graph_features = build_graph_features(transactions, customers["customer_id"].tolist())

# ============================================================
# DEMO PROPENSITY
# ============================================================
@st.cache_data
def build_demo_recommendations(future_targets, customers, products):
    df = future_targets.merge(customers[["customer_id", "segment", "segment_name"]], on="customer_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    product_base = {"P001": .78, "P002": .56, "P003": .45, "P004": .48, "P005": .72, "P006": .52, "P007": .60, "P008": .66, "P009": .47, "P010": .51}
    segment_adjustment = {0: .08, 1: .03, 2: 0.00, 3: .06}
    rng = np.random.default_rng(42)
    df["propensity"] = (df["product_id"].map(product_base) + df["segment"].map(segment_adjustment).fillna(0) + rng.normal(0, .055, len(df))).clip(.05, .95)
    return df

recommendations = build_demo_recommendations(future_targets, customers, products)
owned_products = customer_products.groupby("customer_id")["product_id"].apply(set).to_dict()
recommendations["already_owned"] = recommendations.apply(lambda r: r["product_id"] in owned_products.get(r["customer_id"], set()), axis=1)
recommendation_candidates = recommendations[~recommendations["already_owned"]].copy()

# ============================================================
# HELPERS
# ============================================================
def rupiah_billions(x):
    return f"Rp {x/1e9:.1f}B"

def section(title, caption=None):
    cap = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(f'<div class="section"><div class="section-title">{title}</div>{cap}</div>', unsafe_allow_html=True)

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def panel_open(title, subtitle=""):
    sub = f'<div class="panel-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>{sub}', unsafe_allow_html=True)

def panel_close():
    st.markdown('</div>', unsafe_allow_html=True)

def polish_fig(fig, height=390, margin=None):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=margin or dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="#24364B", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showline=False, gridcolor="#EEF2F6", zeroline=False)
    fig.update_yaxes(showline=False, gridcolor="#EEF2F6", zeroline=False)
    return fig

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown('<div class="brand"><div class="brand-mark">BNI <span>Insight</span></div><div class="brand-sub">Customer Opportunity Intelligence</div></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Workspace", ["Overview", "Customer Segmentation", "Product Opportunity", "Customer 360", "Network Analysis"])
st.sidebar.markdown("<div class='sidebar-footer'>Prototype • Synthetic Business Banking Data</div>", unsafe_allow_html=True)

# ============================================================
# GLOBAL HERO
# ============================================================
st.markdown(
    '<div class="hero"><div class="hero-kicker">Business Banking Analytics</div>'
    '<div class="hero-title">Customer Opportunity Intelligence</div>'
    '<div class="hero-sub">Identify customer segments, understand transaction relationships, and surface relevant product opportunities for Relationship Managers.</div></div>',
    unsafe_allow_html=True,
)

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    high_opportunity = recommendation_candidates.groupby("customer_id")["propensity"].max().ge(.70).sum()
    avg_products = customer_products.groupby("customer_id").size().mean()
    total_tx = len(transactions)
    total_value = transactions["amount"].sum()

    cols = st.columns(5)
    vals = [
        ("Customers", f"{len(customers):,}", "Business customers"),
        ("Transactions", f"{total_tx:,}", "Observed transactions"),
        ("Transaction Value", rupiah_billions(total_value), "Total network flow"),
        ("High Opportunity", f"{high_opportunity:,}", "Propensity ≥ 70%"),
        ("Avg Products", f"{avg_products:.1f}", "Products / customer"),
    ]
    for c, (a,b,d) in zip(cols, vals):
        with c: metric_card(a,b,d)

    section("Portfolio overview", "Segmentation and product opportunity at a glance")
    left, right = st.columns([1.45, 1])
    plot_data = customers.merge(customer_features[["customer_id", "total_outgoing_amount"]], on="customer_id", how="left")

    with left:
        panel_open("Customer portfolio map", "Annual revenue vs. transaction flow")
        fig = px.scatter(plot_data, x="annual_revenue", y="total_outgoing_amount", color="segment_name", hover_name="customer_id", log_x=True, log_y=True, labels={"annual_revenue":"Annual Revenue", "total_outgoing_amount":"Outgoing Transaction Value", "segment_name":"Segment"}, color_discrete_sequence=["#F28C28", "#0D6EFD", "#198754", "#7A5AF8"])
        fig.update_traces(marker=dict(size=7, opacity=.62))
        polish_fig(fig, 410)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        panel_close()

    with right:
        panel_open("Segment profile", "Average portfolio characteristics")
        profile = customers.groupby(["segment", "segment_name"]).agg(Customers=("customer_id","count"), Avg_Revenue=("annual_revenue","mean"), Avg_Employees=("employees","mean")).reset_index()
        profile["Avg Revenue (Rp B)"] = profile["Avg_Revenue"] / 1e9
        fig = px.bar(profile.sort_values("Customers"), x="Customers", y="segment_name", orientation="h", color="segment_name", text="Customers", color_discrete_sequence=["#F28C28", "#0D6EFD", "#198754", "#7A5AF8"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        polish_fig(fig, 410, dict(l=10,r=35,t=20,b=10))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        panel_close()

    section("Product opportunity matrix", "Average demo propensity among customers who do not currently own the product")
    matrix = (recommendation_candidates.groupby(["segment_name", "product_name"])["propensity"].mean().unstack().fillna(0) * 100).round(0)
    fig = px.imshow(matrix, text_auto=True, aspect="auto", labels={"color":"Propensity (%)"}, color_continuous_scale=["#EEF3F8", "#F28C28", "#071A33"])
    polish_fig(fig, 390, dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# SEGMENTATION
# ============================================================
elif page == "Customer Segmentation":
    section("Customer Segmentation", "Behavioral and financial clustering using K-Means")
    summary = customers.groupby(["segment","segment_name"]).agg(Customers=("customer_id","count"), Avg_Revenue=("annual_revenue","mean"), Avg_Balance=("avg_balance","mean"), Avg_Employees=("employees","mean")).reset_index()
    summary["Revenue (Rp B)"] = summary["Avg_Revenue"] / 1e9
    summary["Balance (Rp B)"] = summary["Avg_Balance"] / 1e9
    cols = st.columns(4)
    for i, row in summary.sort_values("segment").iterrows():
        with cols[int(row["segment"])]:
            metric_card(row["segment_name"], f"{int(row['Customers']):,}", f"Avg revenue {row['Revenue (Rp B)']:.1f}B")

    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        panel_open("Segment size")
        fig = px.bar(summary, x="segment_name", y="Customers", color="segment_name", text="Customers", color_discrete_sequence=["#F28C28", "#0D6EFD", "#198754", "#7A5AF8"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        polish_fig(fig, 370)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with right:
        panel_open("Portfolio characteristics")
        display = summary[["segment_name","Customers","Revenue (Rp B)","Balance (Rp B)","Avg_Employees"]].rename(columns={"segment_name":"Segment","Avg_Employees":"Avg Employees"})
        display["Revenue (Rp B)"] = display["Revenue (Rp B)"].round(1)
        display["Balance (Rp B)"] = display["Balance (Rp B)"].round(1)
        st.dataframe(display, hide_index=True, use_container_width=True, height=310)
        panel_close()

# ============================================================
# PRODUCT OPPORTUNITY
# ============================================================
elif page == "Product Opportunity":
    section("Product Opportunity", "Where the portfolio has the strongest cross-sell potential")
    summary = recommendation_candidates.groupby("product_name").agg(Avg_Propensity=("propensity","mean"), Customers=("customer_id","nunique")).reset_index().sort_values("Avg_Propensity", ascending=False)
    summary["Avg_Propensity"] *= 100
    left, right = st.columns([1.3, .9])
    with left:
        panel_open("Product opportunity", "Average propensity across eligible customers")
        fig = px.bar(summary, x="Avg_Propensity", y="product_name", orientation="h", text="Avg_Propensity", color="Avg_Propensity", color_continuous_scale=["#DCE8F3", "#F28C28", "#071A33"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        polish_fig(fig, 460, dict(l=10,r=55,t=20,b=10)); fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with right:
        panel_open("Opportunity summary", "Eligible customers by product")
        top3 = summary.head(5).copy()
        top3["Avg Propensity"] = top3["Avg_Propensity"].round(1).astype(str) + "%"
        top3 = top3.rename(columns={"product_name":"Product","Customers":"Customers"})[["Product","Avg Propensity","Customers"]]
        st.dataframe(top3, hide_index=True, use_container_width=True, height=300)
        panel_close()

    section("Top customer opportunities", "One strongest new-product opportunity per customer")
    top = recommendation_candidates.sort_values(["customer_id","propensity"], ascending=[True,False]).groupby("customer_id").head(1)
    display = top.merge(customers[["customer_id","industry","region","annual_revenue","segment_name"]], on="customer_id", how="left")
    display["Revenue"] = display["annual_revenue"].apply(rupiah_billions)
    display["Propensity"] = (display["propensity"]*100).round(1).astype(str) + "%"
    display = display[["customer_id","segment_name","industry","region","Revenue","product_name","Propensity"]].rename(columns={"customer_id":"Customer","segment_name":"Segment","product_name":"Recommended Product"})
    st.dataframe(display, hide_index=True, use_container_width=True, height=470)

# ============================================================
# CUSTOMER 360
# ============================================================
elif page == "Customer 360":
    section("Customer 360", "Individual relationship view for Relationship Manager decision support")
    customer_id = st.selectbox("Select customer", sorted(customers["customer_id"].unique()))
    customer = customers[customers["customer_id"] == customer_id].iloc[0]
    cf = customer_features[customer_features["customer_id"] == customer_id].iloc[0]

    st.markdown(f'<span class="tag">{customer["segment_name"]}</span> &nbsp; <b>{customer_id}</b> &nbsp; {customer["industry"]} · {customer["region"]}', unsafe_allow_html=True)
    st.write("")
    cols = st.columns(5)
    for c, (label, value, note) in zip(cols, [
        ("Annual Revenue", rupiah_billions(customer["annual_revenue"]), "Company scale"),
        ("Avg Balance", rupiah_billions(customer["avg_balance"]), "Average balance"),
        ("Employees", f"{int(customer['employees']):,}", "Business size"),
        ("Transaction Value", rupiah_billions(cf["total_outgoing_amount"] + cf["total_incoming_amount"]), "Inbound + outbound"),
        ("Counterparties", f"{int(cf['unique_outgoing_counterparties'] + cf['unique_incoming_counterparties']):,}", "Network reach"),
    ]):
        with c: metric_card(label, value, note)

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        panel_open("Existing products", "Products already held by the customer")
        owned_ids = owned_products.get(customer_id, set())
        owned = products[products["product_id"].isin(owned_ids)][["product_name","category"]]
        st.dataframe(owned, hide_index=True, use_container_width=True, height=235)
        panel_close()
    with right:
        panel_open("Recommended next products", "Demo propensity — replace with trained model for final capstone")
        recs = recommendation_candidates[recommendation_candidates["customer_id"] == customer_id].sort_values("propensity", ascending=False).head(3)
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            st.markdown(f'<div class="rec-card"><div class="rec-rank">RECOMMENDATION {rank}</div><div class="rec-name">{row["product_name"]}</div><div class="rec-score">Propensity score: <b>{row["propensity"]:.0%}</b></div></div>', unsafe_allow_html=True)
        panel_close()

    section("Transaction network", "Selected customer and connected counterparties")
    neighbors = list(dict.fromkeys(list(G.predecessors(customer_id)) + list(G.successors(customer_id))))[:15]
    nodes = [customer_id] + neighbors
    subgraph = G.subgraph(nodes).copy()
    pos = nx.spring_layout(subgraph, seed=42)
    edge_x, edge_y = [], []
    for u,v in subgraph.edges():
        x0,y0=pos[u]; x1,y1=pos[v]; edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    edge_trace = go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=1,color="#CBD5E1"),hoverinfo="none")
    node_x,node_y,node_text,node_color=[],[],[],[]
    for node in nodes:
        x,y=pos[node]; node_x.append(x); node_y.append(y); node_text.append(f"{node}<br>{'Selected Customer' if node==customer_id else 'Counterparty'}"); node_color.append("#F28C28" if node==customer_id else "#0D6EFD")
    node_trace=go.Scatter(x=node_x,y=node_y,mode="markers+text",text=[n if n==customer_id else "" for n in nodes],textposition="bottom center",hovertext=node_text,hoverinfo="text",marker=dict(size=[24 if n==customer_id else 14 for n in nodes],color=node_color,line=dict(width=2,color="#FFFFFF")))
    fig=go.Figure([edge_trace,node_trace]); fig.update_layout(showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=500)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ============================================================
# NETWORK ANALYSIS
# ============================================================
elif page == "Network Analysis":
    section("Transaction Network Analysis", "Relationship structure across the business customer ecosystem")
    cols=st.columns(4)
    for c,(label,value,note) in zip(cols,[
        ("Nodes",f"{G.number_of_nodes():,}","Business customers"),
        ("Edges",f"{G.number_of_edges():,}","Observed relationships"),
        ("Density",f"{nx.density(G):.5f}","Network connectivity"),
        ("Total Flow",rupiah_billions(transactions["amount"].sum()),"Transaction value"),
    ]):
        with c: metric_card(label,value,note)

    section("Core transaction network", "Top 30 customers by outgoing transaction value")
    top_nodes = transactions.groupby("sender_customer_id")["amount"].sum().nlargest(30).index.tolist()
    subgraph = G.subgraph(top_nodes).copy()
    pos = nx.spring_layout(subgraph, seed=42, k=1.5)
    edge_x,edge_y=[],[]
    for u,v in subgraph.edges():
        x0,y0=pos[u];x1,y1=pos[v];edge_x += [x0,x1,None];edge_y += [y0,y1,None]
    edge_trace=go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=.8,color="#D4DCE5"),hoverinfo="none")
    node_x,node_y,node_text=[],[],[]
    for node in subgraph.nodes():
        x,y=pos[node];node_x.append(x);node_y.append(y);node_text.append(node)
    node_trace=go.Scatter(x=node_x,y=node_y,mode="markers+text",text=node_text,textposition="top center",hoverinfo="text",marker=dict(size=15,color="#F28C28",line=dict(width=2,color="#FFFFFF")))
    fig=go.Figure([edge_trace,node_trace]);fig.update_layout(showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=650)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
