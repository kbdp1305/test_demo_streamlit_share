
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
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    required_files = [
        "customers.csv",
        "products.csv",
        "customer_products.csv",
        "transactions.csv",
        "monthly_customer_behavior.csv",
        "customer_feature_table.csv",
        "future_product_targets.csv",
    ]

    missing = [
        f for f in required_files
        if not (DATA_DIR / f).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files in data/: " + ", ".join(missing)
        )

    customers = pd.read_csv(
        DATA_DIR / "customers.csv"
    )

    products = pd.read_csv(
        DATA_DIR / "products.csv"
    )

    customer_products = pd.read_csv(
        DATA_DIR / "customer_products.csv"
    )

    transactions = pd.read_csv(
        DATA_DIR / "transactions.csv"
    )

    monthly_behavior = pd.read_csv(
        DATA_DIR / "monthly_customer_behavior.csv"
    )

    customer_features = pd.read_csv(
        DATA_DIR / "customer_feature_table.csv"
    )

    future_targets = pd.read_csv(
        DATA_DIR / "future_product_targets.csv"
    )

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    )

    return (
        customers,
        products,
        customer_products,
        transactions,
        monthly_behavior,
        customer_features,
        future_targets
    )


try:

    (
        customers,
        products,
        customer_products,
        transactions,
        monthly_behavior,
        customer_features,
        future_targets
    ) = load_data()

except Exception as e:

    st.error(str(e))

    st.info(
        "Pastikan 7 file CSV berada di folder data/ "
        "dalam repository."
    )

    st.stop()


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

@st.cache_data
def build_segmentation(customer_features):

    feature_cols = [
        "annual_revenue",
        "avg_balance",
        "employees",
        "total_outgoing_amount",
        "total_outgoing_tx",
        "avg_outgoing_amount",
        "unique_outgoing_counterparties",
        "total_incoming_amount",
        "total_incoming_tx",
        "avg_incoming_amount",
        "unique_incoming_counterparties",
        "product_count",
    ]

    X = customer_features[
        feature_cols
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # Log transform because banking monetary features
    # are highly right-skewed.
    X_log = np.log1p(X)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_log
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    result = customer_features[
        ["customer_id"]
    ].copy()

    result["segment"] = labels

    return result


segment_result = build_segmentation(
    customer_features
)


customers = customers.merge(
    segment_result,
    on="customer_id",
    how="left"
)


segment_names = {
    0: "High Value Business",
    1: "Transactional Business",
    2: "Emerging Business",
    3: "Network Connected"
}

customers["segment_name"] = (
    customers["segment"]
    .map(segment_names)
)


# ============================================================
# TRANSACTION GRAPH
# ============================================================

@st.cache_resource
def build_graph(transactions, customer_ids):

    graph = nx.DiGraph()

    graph.add_nodes_from(
        customer_ids
    )

    edges = (
        transactions
        .groupby(
            [
                "sender_customer_id",
                "receiver_customer_id"
            ]
        )
        .agg(
            transaction_count=(
                "transaction_id",
                "count"
            ),
            total_amount=(
                "amount",
                "sum"
            )
        )
        .reset_index()
    )

    for _, row in edges.iterrows():

        graph.add_edge(
            row["sender_customer_id"],
            row["receiver_customer_id"],
            transaction_count=row["transaction_count"],
            total_amount=row["total_amount"]
        )

    return graph


G = build_graph(
    transactions,
    customers["customer_id"].tolist()
)


# ============================================================
# GRAPH FEATURES
# ============================================================

@st.cache_data
def build_graph_features(
    transactions,
    customer_ids
):

    graph = nx.DiGraph()

    graph.add_nodes_from(
        customer_ids
    )

    edges = (
        transactions
        .groupby(
            [
                "sender_customer_id",
                "receiver_customer_id"
            ]
        )
        .agg(
            transaction_count=(
                "transaction_id",
                "count"
            ),
            total_amount=(
                "amount",
                "sum"
            )
        )
        .reset_index()
    )

    for _, row in edges.iterrows():

        graph.add_edge(
            row["sender_customer_id"],
            row["receiver_customer_id"],
            total_amount=row["total_amount"]
        )

    pagerank = nx.pagerank(
        graph,
        weight="total_amount"
    )

    result = pd.DataFrame({
        "customer_id": customer_ids,

        "in_degree": [
            graph.in_degree(c)
            for c in customer_ids
        ],

        "out_degree": [
            graph.out_degree(c)
            for c in customer_ids
        ],

        "pagerank": [
            pagerank.get(c, 0)
            for c in customer_ids
        ]
    })

    return result


graph_features = build_graph_features(
    transactions,
    customers["customer_id"].tolist()
)


# ============================================================
# DEMO PRODUCT PROPENSITY
# ============================================================
# This is intentionally a DEMO score.
# Final capstone: replace this section with
# model.predict_proba() from the trained model.

@st.cache_data
def build_demo_recommendations(
    future_targets,
    customers,
    products
):

    df = future_targets.merge(
        customers[
            [
                "customer_id",
                "segment",
                "segment_name"
            ]
        ],
        on="customer_id",
        how="left"
    )

    df = df.merge(
        products,
        on="product_id",
        how="left"
    )

    product_base = {
        "P001": 0.78,
        "P002": 0.56,
        "P003": 0.45,
        "P004": 0.48,
        "P005": 0.72,
        "P006": 0.52,
        "P007": 0.60,
        "P008": 0.66,
        "P009": 0.47,
        "P010": 0.51,
    }

    segment_adjustment = {
        0: 0.08,
        1: 0.03,
        2: 0.00,
        3: 0.06,
    }

    # Stable deterministic noise for demo.
    rng = np.random.default_rng(42)

    df["propensity"] = (
        df["product_id"].map(product_base)
        + df["segment"].map(segment_adjustment).fillna(0)
        + rng.normal(
            0,
            0.055,
            len(df)
        )
    ).clip(0.05, 0.95)

    return df


recommendations = build_demo_recommendations(
    future_targets,
    customers,
    products
)


# ============================================================
# EXISTING PRODUCT FILTER
# ============================================================

owned_products = (
    customer_products
    .groupby("customer_id")["product_id"]
    .apply(set)
    .to_dict()
)

recommendations["already_owned"] = recommendations.apply(
    lambda row:
        row["product_id"]
        in owned_products.get(
            row["customer_id"],
            set()
        ),
    axis=1
)

recommendation_candidates = recommendations[
    ~recommendations["already_owned"]
].copy()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 BNI Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Customer Segmentation",
        "Product Opportunity",
        "Customer 360",
        "Network Analysis",
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Customer Opportunity Intelligence"
)

st.sidebar.caption(
    "Streamlit Demo"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'BNI Customer Opportunity Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Data-driven customer segmentation, transaction network analysis, '
    'and product opportunity recommendation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    high_opportunity = (
        recommendation_candidates
        .groupby("customer_id")["propensity"]
        .max()
        .ge(0.70)
        .sum()
    )

    avg_products = (
        customer_products
        .groupby("customer_id")
        .size()
        .mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{len(customers):,}"
    )

    c2.metric(
        "High Opportunity",
        f"{high_opportunity:,}"
    )

    c3.metric(
        "Avg Products / Customer",
        f"{avg_products:.1f}"
    )

    c4.metric(
        "Customer Segments",
        f"{customers['segment'].nunique()}"
    )

    st.markdown(
        '<div class="section-title">'
        'Customer Segmentation'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.4, 1]
    )

    plot_data = customers.merge(
        customer_features[
            [
                "customer_id",
                "total_outgoing_amount"
            ]
        ],
        on="customer_id",
        how="left"
    )

    with left:

        fig = px.scatter(
            plot_data,
            x="annual_revenue",
            y="total_outgoing_amount",
            color="segment_name",
            hover_name="customer_id",
            log_x=True,
            log_y=True,
            labels={
                "annual_revenue":
                    "Annual Revenue (Rp)",
                "total_outgoing_amount":
                    "Total Outgoing (Rp)",
                "segment_name":
                    "Segment"
            }
        )

        fig.update_layout(
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        profile = (
            customers
            .groupby(
                [
                    "segment",
                    "segment_name"
                ]
            )
            .agg(
                Customers=(
                    "customer_id",
                    "count"
                ),
                Avg_Revenue=(
                    "annual_revenue",
                    "mean"
                ),
                Avg_Employees=(
                    "employees",
                    "mean"
                )
            )
            .reset_index()
        )

        profile[
            "Avg Revenue (Rp B)"
        ] = (
            profile["Avg_Revenue"]
            / 1e9
        ).round(1)

        st.dataframe(
            profile[
                [
                    "segment_name",
                    "Customers",
                    "Avg Revenue (Rp B)",
                    "Avg_Employees"
                ]
            ].rename(
                columns={
                    "segment_name":
                        "Segment",
                    "Avg_Employees":
                        "Avg Employees"
                }
            ),
            hide_index=True,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">'
        'Product Opportunity by Segment'
        '</div>',
        unsafe_allow_html=True
    )

    matrix = (
        recommendation_candidates
        .groupby(
            [
                "segment_name",
                "product_name"
            ]
        )["propensity"]
        .mean()
        .unstack()
        .fillna(0)
    )

    matrix = (
        matrix * 100
    ).round(0)

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        labels={
            "color":
                "Avg Propensity (%)"
        }
    )

    fig.update_layout(
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SEGMENTATION
# ============================================================

elif page == "Customer Segmentation":

    st.header(
        "Customer Segmentation"
    )

    summary = (
        customers
        .groupby(
            [
                "segment",
                "segment_name"
            ]
        )
        .agg(
            Customers=(
                "customer_id",
                "count"
            ),
            Avg_Revenue=(
                "annual_revenue",
                "mean"
            ),
            Avg_Balance=(
                "avg_balance",
                "mean"
            ),
            Avg_Employees=(
                "employees",
                "mean"
            )
        )
        .reset_index()
    )

    summary["Avg Revenue (Rp B)"] = (
        summary["Avg_Revenue"] / 1e9
    ).round(2)

    summary["Avg Balance (Rp B)"] = (
        summary["Avg_Balance"] / 1e9
    ).round(2)

    st.dataframe(
        summary[
            [
                "segment_name",
                "Customers",
                "Avg Revenue (Rp B)",
                "Avg Balance (Rp B)",
                "Avg_Employees"
            ]
        ].rename(
            columns={
                "segment_name":
                    "Segment",
                "Avg_Employees":
                    "Avg Employees"
            }
        ),
        hide_index=True,
        use_container_width=True
    )

    fig = px.bar(
        summary,
        x="segment_name",
        y="Customers",
        color="segment_name",
        labels={
            "segment_name":
                "Segment",
            "Customers":
                "Number of Customers"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PRODUCT OPPORTUNITY
# ============================================================

elif page == "Product Opportunity":

    st.header(
        "Product Opportunity"
    )

    summary = (
        recommendation_candidates
        .groupby("product_name")
        .agg(
            Avg_Propensity=(
                "propensity",
                "mean"
            ),
            Customers=(
                "customer_id",
                "nunique"
            )
        )
        .reset_index()
        .sort_values(
            "Avg_Propensity",
            ascending=False
        )
    )

    summary["Avg_Propensity"] *= 100

    fig = px.bar(
        summary.head(10),
        x="Avg_Propensity",
        y="product_name",
        orientation="h",
        text="Avg_Propensity",
        labels={
            "Avg_Propensity":
                "Average Propensity (%)",
            "product_name":
                "Product"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Top Customer Opportunities"
    )

    top = (
        recommendation_candidates
        .sort_values(
            [
                "customer_id",
                "propensity"
            ],
            ascending=[
                True,
                False
            ]
        )
        .groupby("customer_id")
        .head(1)
    )

    display = top.merge(
        customers[
            [
                "customer_id",
                "industry",
                "region",
                "annual_revenue",
                "segment_name"
            ]
        ],
        on="customer_id",
        how="left"
    )

    display["Revenue (Rp B)"] = (
        display["annual_revenue"] / 1e9
    ).round(1)

    display["Propensity"] = (
        display["propensity"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(
        display[
            [
                "customer_id",
                "segment_name",
                "industry",
                "region",
                "Revenue (Rp B)",
                "product_name",
                "Propensity"
            ]
        ].rename(
            columns={
                "customer_id":
                    "Customer ID",
                "segment_name":
                    "Segment",
                "industry":
                    "Industry",
                "region":
                    "Region",
                "product_name":
                    "Top Recommendation"
            }
        ),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# CUSTOMER 360
# ============================================================

elif page == "Customer 360":

    st.header(
        "Customer 360"
    )

    customer_id = st.selectbox(
        "Select Customer",
        sorted(
            customers[
                "customer_id"
            ].unique()
        )
    )

    customer = customers[
        customers["customer_id"]
        == customer_id
    ].iloc[0]

    cf = customer_features[
        customer_features[
            "customer_id"
        ] == customer_id
    ].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Industry",
        customer["industry"]
    )

    c2.metric(
        "Revenue",
        f"Rp {customer['annual_revenue']/1e9:.1f}B"
    )

    c3.metric(
        "Balance",
        f"Rp {customer['avg_balance']/1e9:.1f}B"
    )

    c4.metric(
        "Employees",
        f"{int(customer['employees']):,}"
    )

    c5.metric(
        "Segment",
        customer["segment_name"]
    )

    st.subheader(
        "Transaction Behavior"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Outgoing",
        f"Rp {cf['total_outgoing_amount']/1e9:.2f}B"
    )

    c2.metric(
        "Incoming",
        f"Rp {cf['total_incoming_amount']/1e9:.2f}B"
    )

    c3.metric(
        "Transactions",
        f"{int(cf['total_outgoing_tx'] + cf['total_incoming_tx']):,}"
    )

    c4.metric(
        "Counterparties",
        f"{int(cf['unique_outgoing_counterparties'] + cf['unique_incoming_counterparties']):,}"
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Existing Products"
        )

        owned_ids = owned_products.get(
            customer_id,
            set()
        )

        owned = products[
            products["product_id"]
            .isin(owned_ids)
        ]

        st.dataframe(
            owned[
                [
                    "product_name",
                    "category"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Top Recommendations"
        )

        recs = (
            recommendation_candidates[
                recommendation_candidates[
                    "customer_id"
                ] == customer_id
            ]
            .sort_values(
                "propensity",
                ascending=False
            )
            .head(3)
        )

        if len(recs) == 0:

            st.info(
                "No new product recommendation "
                "available for this customer."
            )

        for rank, (_, row) in enumerate(
            recs.iterrows(),
            start=1
        ):

            st.markdown(
                f"**{rank}. {row['product_name']}**"
            )

            st.progress(
                float(row["propensity"]),
                text=(
                    f"{row['propensity']:.0%} "
                    "propensity"
                )
            )

    st.subheader(
        "Customer Transaction Network"
    )

    neighbors = list(
        G.predecessors(customer_id)
    ) + list(
        G.successors(customer_id)
    )

    neighbors = list(
        dict.fromkeys(
            neighbors
        )
    )[:15]

    nodes = [
        customer_id
    ] + neighbors

    subgraph = G.subgraph(
        nodes
    ).copy()

    pos = nx.spring_layout(
        subgraph,
        seed=42
    )

    edge_x = []
    edge_y = []

    for u, v in subgraph.edges():

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend(
            [x0, x1, None]
        )

        edge_y.extend(
            [y0, y1, None]
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none"
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in nodes:

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)

        if node == customer_id:

            node_color.append(
                "red"
            )

            node_text.append(
                f"{node}<br>Selected Customer"
            )

        else:

            node_color.append(
                "steelblue"
            )

            node_text.append(
                f"{node}<br>Counterparty"
            )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[
            n if n == customer_id else ""
            for n in nodes
        ],
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=18,
            color=node_color
        )
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace
        ]
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            visible=False
        ),
        yaxis=dict(
            showgrid=False,
            visible=False
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# NETWORK ANALYSIS
# ============================================================

elif page == "Network Analysis":

    st.header(
        "Transaction Network Analysis"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Nodes",
        f"{G.number_of_nodes():,}"
    )

    c2.metric(
        "Edges",
        f"{G.number_of_edges():,}"
    )

    c3.metric(
        "Network Density",
        f"{nx.density(G):.5f}"
    )

    top_nodes = (
        transactions
        .groupby(
            "sender_customer_id"
        )["amount"]
        .sum()
        .nlargest(30)
        .index
        .tolist()
    )

    subgraph = G.subgraph(
        top_nodes
    ).copy()

    pos = nx.spring_layout(
        subgraph,
        seed=42,
        k=1.5
    )

    edge_x = []
    edge_y = []

    for u, v in subgraph.edges():

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend(
            [x0, x1, None]
        )

        edge_y.extend(
            [y0, y1, None]
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.7),
        hoverinfo="none"
    )

    node_x = []
    node_y = []
    node_text = []

    for node in subgraph.nodes():

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=14
        )
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace
        ]
    )

    fig.update_layout(
        height=650,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            visible=False
        ),
        yaxis=dict(
            showgrid=False,
            visible=False
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
