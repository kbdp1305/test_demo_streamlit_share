import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title='BNI Customer Opportunity Intelligence', page_icon='🏦', layout='wide', initial_sidebar_state='expanded')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

# -----------------------------
# Premium dashboard styling
# -----------------------------
st.markdown('''
<style>
:root{--navy:#071a33;--navy2:#0d2948;--orange:#f28c28;--blue:#2f80ed;--green:#39b86f;--purple:#8267e8;--ink:#10233f;--muted:#6b7a90;--line:#e5ebf2;--bg:#f4f7fa;--white:#fff}
.stApp{background:var(--bg)}
.block-container{max-width:1520px;padding:1rem 1.2rem 2rem}
[data-testid='stHeader']{background:rgba(244,247,250,.96)}
[data-testid='stSidebar']{background:linear-gradient(180deg,#06172c,#0b2948);border:0}
[data-testid='stSidebar'] *{color:#eaf1f8!important}
[data-testid='stSidebar'] .stRadio>label{display:none}
[data-testid='stSidebar'] .stRadio div[role='radiogroup']{gap:.3rem}
[data-testid='stSidebar'] .stRadio div[role='radiogroup'] label{padding:.62rem .72rem;border-radius:.65rem;font-weight:700;font-size:.78rem}
[data-testid='stSidebar'] .stRadio div[role='radiogroup'] label:hover{background:rgba(255,255,255,.08)}
[data-testid='stSidebar'] [data-baseweb='radio']>div:first-child{display:none}
[data-testid='stSidebar'] .stRadio div[role='radiogroup'] label:has(input:checked){background:#0d6fb9;border-left:3px solid #ff9e32}
.brand{padding:.2rem .25rem 1.2rem}.brand-row{display:flex;align-items:center;gap:.6rem}.brand-symbol{width:42px;height:42px;border-radius:10px;background:#f28c28;display:flex;align-items:center;justify-content:center;color:white;font-size:1.4rem;font-weight:900}.brand-name{font-size:1.55rem;font-weight:900;letter-spacing:-.04em}.brand-sub{font-size:.68rem;color:#9eb2c7!important;margin-top:.15rem}
.sidebar-section{font-size:.64rem;color:#88a0b8!important;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin:.8rem 0 .35rem}
.hero{background:linear-gradient(115deg,#061a32,#12395c 70%,#194b70);border-radius:16px;padding:1.25rem 1.35rem;color:#fff;box-shadow:0 8px 25px rgba(7,26,51,.1);margin-bottom:.7rem}.hero-title{font-size:1.95rem;font-weight:900;letter-spacing:-.045em;line-height:1.1}.hero-sub{color:#b9ccdf;font-size:.84rem;margin-top:.25rem}.hero-kicker{color:#ffad52;font-size:.63rem;font-weight:900;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.3rem}
.filter-card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.45rem .55rem;margin-bottom:.7rem}.filter-card [data-testid='stSelectbox'] label{font-size:.62rem;font-weight:800;color:#51627a;margin-bottom:.12rem}.filter-card [data-baseweb='select']{min-height:34px}
.card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:.85rem .95rem;box-shadow:0 3px 14px rgba(16,35,63,.035)}
.kpi{display:flex;gap:.75rem;align-items:center}.kpi-icon{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.1rem;background:#edf5ff}.kpi-label{font-size:.65rem;color:#6b7a90;font-weight:800}.kpi-value{font-size:1.42rem;color:#10233f;font-weight:900;letter-spacing:-.035em;line-height:1.1}.kpi-note{font-size:.62rem;color:#3aaf70;margin-top:.18rem}
.section-title{font-size:.98rem;font-weight:900;color:#10233f;letter-spacing:-.02em}.section-sub{font-size:.65rem;color:#7a8798;margin-top:.12rem;margin-bottom:.45rem}
.seg-table{width:100%;border-collapse:collapse;font-size:.68rem}.seg-table th{text-align:left;color:#6c7a8d;font-size:.6rem;padding:.4rem;border-bottom:1px solid #edf0f4}.seg-table td{padding:.48rem .4rem;border-bottom:1px solid #f0f2f5;color:#223650}.seg-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.seg-name{font-weight:800}.small-note{font-size:.61rem;color:#7c8b9d}
.rec{border:1px solid #e8edf3;border-radius:9px;padding:.55rem .6rem;margin-bottom:.42rem;background:#fff}.rec-head{display:flex;align-items:center;gap:.45rem}.rec-rank{width:22px;height:22px;border-radius:6px;background:#fff1df;color:#b7610b;font-weight:900;font-size:.68rem;display:flex;align-items:center;justify-content:center}.rec-name{font-size:.74rem;font-weight:900;color:#122a45}.rec-score{font-size:.66rem;color:#6d7d91}.bar{height:6px;border-radius:9px;background:#edf1f5;overflow:hidden;margin-top:.3rem}.bar>div{height:100%;border-radius:9px;background:linear-gradient(90deg,#34b96f,#54c77f)}
.info-title{font-size:.65rem;color:#738196;font-weight:800}.info-value{font-size:.75rem;color:#1b304a;font-weight:800;margin-top:.08rem}.tag{display:inline-block;padding:.22rem .5rem;border-radius:999px;background:#eaf8ef;color:#16804a;font-size:.61rem;font-weight:900}.tag-blue{background:#eaf3ff;color:#2368b7}.tag-orange{background:#fff1df;color:#b7610b}
[data-testid='stDataFrame']{border-radius:10px;overflow:hidden}.stPlotlyChart{margin-top:-.1rem}
hr{border:0;border-top:1px solid #e9edf2;margin:.7rem 0}
</style>
''', unsafe_allow_html=True)

@st.cache_data

def load_data():
    customers = pd.read_csv(DATA_DIR/'customers.csv')
    products = pd.read_csv(DATA_DIR/'products.csv')
    customer_products = pd.read_csv(DATA_DIR/'customer_products.csv')
    transactions = pd.read_csv(DATA_DIR/'transactions.csv', parse_dates=['transaction_date'])
    features = pd.read_csv(DATA_DIR/'customer_feature_table.csv')
    monthly = pd.read_csv(DATA_DIR/'monthly_customer_behavior.csv', parse_dates=['month'])
    targets = pd.read_csv(DATA_DIR/'future_product_targets.csv')
    return customers, products, customer_products, transactions, features, monthly, targets

customers, products, customer_products, transactions, features, monthly, targets = load_data()

# -----------------------------
# Customer/product analytics
# -----------------------------
@st.cache_data

def build_segments(features):
    f = features.copy()
    numeric = ['annual_revenue','avg_balance','employees','customer_age_years','total_outgoing_amount','total_outgoing_tx','avg_outgoing_amount','unique_outgoing_counterparties','total_incoming_amount','total_incoming_tx','avg_incoming_amount','unique_incoming_counterparties','product_count']
    X = f[numeric].replace([np.inf,-np.inf],np.nan).fillna(0)
    Xs = StandardScaler().fit_transform(np.log1p(X))
    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    f['segment_id'] = model.fit_predict(Xs)
    prof = f.groupby('segment_id').agg(customers=('customer_id','count'), avg_revenue=('annual_revenue','mean'), avg_tx=('total_outgoing_amount','mean'), avg_counterparty=('unique_outgoing_counterparties','mean'), avg_products=('product_count','mean')).reset_index()
    # Give readable names based on revenue/transaction profile.
    prof['score'] = prof['avg_revenue'].rank(method='first')
    names = {}
    # deterministic semantic ordering by revenue
    ordered = prof.sort_values('avg_revenue')['segment_id'].tolist()
    semantic = ['Emerging Business','Transactional Business','Network Connected','High Value Business']
    for sid,name in zip(ordered,semantic): names[sid]=name
    f['segment'] = f['segment_id'].map(names)
    return f, prof, names

seg_features, seg_profile_raw, segment_names = build_segments(features)

@st.cache_data

def graph_features(transactions):
    G = nx.DiGraph()
    for r in transactions.itertuples(index=False):
        s, t, a = r.sender_customer_id, r.receiver_customer_id, float(r.amount)
        if G.has_edge(s,t): G[s][t]['weight'] += a; G[s][t]['count'] += 1
        else: G.add_edge(s,t,weight=a,count=1)
    out_amt = dict(G.out_degree(weight='weight')); in_amt=dict(G.in_degree(weight='weight'))
    out_deg=dict(G.out_degree()); in_deg=dict(G.in_degree())
    pr = nx.pagerank(G, weight='weight') if len(G) else {}
    # Full betweenness can be expensive; approximate on this large graph.
    btw = nx.betweenness_centrality(G, k=min(120,len(G)), weight=None, seed=42) if len(G)>120 else nx.betweenness_centrality(G, weight=None)
    gf = pd.DataFrame({'customer_id':list(G.nodes()),'out_degree':[out_deg.get(x,0) for x in G.nodes()],'in_degree':[in_deg.get(x,0) for x in G.nodes()],'out_amount':[out_amt.get(x,0) for x in G.nodes()],'in_amount':[in_amt.get(x,0) for x in G.nodes()],'pagerank':[pr.get(x,0) for x in G.nodes()],'betweenness':[btw.get(x,0) for x in G.nodes()]})
    return G,gf

@st.cache_data

def recommendation_scores(features, products, customer_products, seg_features):
    base = seg_features[['customer_id','segment']].merge(features,on='customer_id',suffixes=('','_f'))
    owned = customer_products.groupby('customer_id')['product_id'].apply(set).to_dict()
    # Synthetic demo propensity, designed to be deterministic and interpretable.
    rows=[]
    revenue_log=np.log1p(base['annual_revenue']); tx_log=np.log1p(base['total_outgoing_amount']); cp=np.log1p(base['unique_outgoing_counterparties'])
    revn=(revenue_log-revenue_log.min())/(revenue_log.max()-revenue_log.min()+1e-9)
    txn=(tx_log-tx_log.min())/(tx_log.max()-tx_log.min()+1e-9)
    cpn=(cp-cp.min())/(cp.max()-cp.min()+1e-9)
    for i,p in products.iterrows():
        name=p.product_name
        if name=='Working Capital Loan': s=.30*txn+.25*revn+.15*cpn
        elif name=='Cash Management': s=.32*txn+.18*cpn+.10*(base.product_count<3)
        elif name=='Trade Finance': s=.28*revn+.25*cpn+.08*(base.industry.isin(['Manufacturing','Wholesale','Logistics']))
        elif name=='Virtual Account': s=.25*txn+.22*cpn+.10*(base.industry.isin(['Retail','Technology']))
        elif name=='Payroll Service': s=.15*base.employees.rank(pct=True)+.20*revn+.08*(base.product_count<4)
        elif name=='Business Current Account': s=.18*txn+.12*(base.product_count<2)+.10*revn
        elif name=='Time Deposit': s=.20*base.avg_balance.rank(pct=True)+.08*revn
        elif name=='Investment Loan': s=.25*revn+.14*base.avg_balance.rank(pct=True)
        elif name=='Corporate Credit Card': s=.16*base.employees.rank(pct=True)+.12*revn+.05*(base.product_count<4)
        else: s=.18*cpn+.12*revn
        # tiny deterministic segment/industry modifiers
        s = np.asarray(s,dtype=float)
        s += np.where(base.segment.str.contains('Network'),.025,0)
        score = np.clip(.28 + .68*s, .12, .96)
        for cid,val in zip(base.customer_id,score):
            if p.product_id not in owned.get(cid,set()): rows.append((cid,p.product_id,name,float(val)))
    return pd.DataFrame(rows,columns=['customer_id','product_id','product_name','propensity'])

@st.cache_data

def make_customer_table(features, seg_features, recs):
    top = recs.sort_values(['customer_id','propensity'],ascending=[True,False]).groupby('customer_id').head(1)
    out=features.merge(seg_features[['customer_id','segment']],on='customer_id').merge(top[['customer_id','product_name','propensity']],on='customer_id',how='left')
    return out

G, gf = graph_features(transactions)
seg_features = seg_features.merge(gf,on='customer_id',how='left').fillna({'out_degree':0,'in_degree':0,'out_amount':0,'in_amount':0,'pagerank':0,'betweenness':0})
recs = recommendation_scores(features,products,customer_products,seg_features)
customer_table = make_customer_table(features,seg_features,recs)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-row"><div class="brand-symbol">✦</div><div><div class="brand-name">BNI</div><div class="brand-sub">Customer Opportunity Intelligence</div></div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Navigation</div>',unsafe_allow_html=True)
    page=st.radio('nav',['Overview','Customer Segmentation','Product Opportunity','Customer 360','Network Analysis'],format_func=lambda x:{'Overview':'⌂  Overview','Customer Segmentation':'◉  Customer Segmentation','Product Opportunity':'▣  Product Opportunity','Customer 360':'♙  Customer 360','Network Analysis':'⌁  Network Analysis'}[x],label_visibility='collapsed')
    st.markdown('<hr>',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Data</div>',unsafe_allow_html=True)
    st.caption(f'{len(customers):,} customers · {len(transactions):,} transactions')
    st.caption('Demo environment · Synthetic data')

# -----------------------------
# Global filters
# -----------------------------
st.markdown('<div class="hero"><div class="hero-kicker">Business Banking Analytics</div><div class="hero-title">BNI Customer Opportunity Intelligence</div><div class="hero-sub">Data-driven product recommendation for Business Banking — combining Customer 360, transaction networks, segmentation and product propensity.</div></div>',unsafe_allow_html=True)

fcols=st.columns([1,1,1,1,1.5])
with fcols[0]: period=st.selectbox('Period',['All','Last 12 Months','Last 6 Months'])
with fcols[1]: seg_filter=st.selectbox('Segment',['All']+sorted(seg_features.segment.unique().tolist()))
with fcols[2]: industry_filter=st.selectbox('Industry',['All']+sorted(customers.industry.unique().tolist()))
with fcols[3]: region_filter=st.selectbox('Region',['All']+sorted(customers.region.unique().tolist()))
with fcols[4]: search=st.text_input('Customer search',placeholder='Search customer ID...')

filtered=customer_table.copy()
if seg_filter!='All': filtered=filtered[filtered.segment==seg_filter]
if industry_filter!='All': filtered=filtered[industry==industry_filter] if False else filtered[filtered.industry==industry_filter]
if region_filter!='All': filtered=filtered[filtered.region==region_filter]
if search: filtered=filtered[filtered.customer_id.str.contains(search,case=False,na=False)]

# -----------------------------
# helpers
# -----------------------------
def money(v):
    v=float(v)
    if abs(v)>=1e9:return f'Rp {v/1e9:.1f} B'
    if abs(v)>=1e6:return f'Rp {v/1e6:.1f} M'
    return f'Rp {v:,.0f}'

def panel(title, subtitle=''):
    st.markdown(f'<div class="section-title">{title}</div>'+ (f'<div class="section-sub">{subtitle}</div>' if subtitle else ''),unsafe_allow_html=True)

def plot_clean(fig,height=300):
    fig.update_layout(height=height,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Inter,Arial',size=10,color='#42556e'),legend=dict(orientation='h',y=1.08,x=0),xaxis=dict(showgrid=False),yaxis=dict(gridcolor='#edf1f5'))
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

# -----------------------------
# Overview — screenshot-inspired
# -----------------------------
if page=='Overview':
    high = int((filtered.propensity>=.72).sum())
    avg_products=filtered.product_count.mean() if len(filtered) else 0
    c1,c2,c3,c4=st.columns(4)
    for col,icon,label,val,note,bg in [
        (c1,'👥','Total Customers',f'{len(filtered):,}','Filtered customer base','#edf5ff'),
        (c2,'🎯','High Opportunity',f'{high:,}','Propensity ≥ 72%','#fff1f1'),
        (c3,'▣','Avg Products / Customer',f'{avg_products:.1f}','Current holdings','#f0f8f4'),
        (c4,'◔','Customer Segments',f'{filtered.segment.nunique() if len(filtered) else 0}','Behavioral clusters','#f4f0ff')]:
        with col:
            st.markdown(f'<div class="card"><div class="kpi"><div class="kpi-icon" style="background:{bg}">{icon}</div><div><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-note">↗ {note}</div></div></div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>',unsafe_allow_html=True)

    left,right=st.columns([1.15,.95])
    with left:
        st.markdown('<div class="card">',unsafe_allow_html=True); panel('Customer Segmentation','Annual revenue vs total outgoing transaction value')
        p=filtered.copy(); p['plot_revenue']=p.annual_revenue.clip(lower=1); p['plot_tx']=p.total_outgoing_amount.clip(lower=1)
        fig=px.scatter(p,x='plot_revenue',y='plot_tx',color='segment',hover_name='customer_id',hover_data=['industry','region','product_count'],log_x=True,log_y=True,color_discrete_sequence=['#2f80ed','#f28c28','#39b86f','#8267e8'])
        fig.update_traces(marker=dict(size=7,opacity=.78))
        fig.update_xaxes(title='Annual Revenue (Rp)',gridcolor='#edf1f5');fig.update_yaxes(title='Total Transaction Value (Rp)',gridcolor='#edf1f5')
        plot_clean(fig,330); st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">',unsafe_allow_html=True); panel('Segment Profile','Behavioral and financial summary')
        sp=seg_features.groupby('segment').agg(Customers=('customer_id','count'),Revenue=('annual_revenue','mean'),Tx=('total_outgoing_amount','mean'),Counterparties=('unique_outgoing_counterparties','mean'),Products=('product_count','mean')).reset_index()
        sp['Revenue']=sp.Revenue.map(money);sp['Tx']=sp.Tx.map(money);sp['Customers']=sp.Customers.map(lambda x:f'{x:,}');sp['Counterparties']=sp.Counterparties.round(0).astype(int);sp['Products']=sp.Products.round(1)
        rows=''.join([f"<tr><td><span class='seg-dot' style='background:{['#2f80ed','#f28c28','#39b86f','#8267e8'][i%4]}'></span><span class='seg-name'>{r.segment}</span></td><td>{r.Customers}</td><td>{r.Revenue}</td><td>{r.Tx}</td><td>{r.Counterparties}</td><td>{r.Products}</td></tr>" for i,r in sp.iterrows()])
        st.markdown(f"<table class='seg-table'><thead><tr><th>Segment</th><th>Customers</th><th>Avg Revenue</th><th>Avg Tx</th><th>Avg Counterparty</th><th>Avg Products</th></tr></thead><tbody>{rows}</tbody></table>",unsafe_allow_html=True);st.markdown('</div>',unsafe_allow_html=True)

    a,b=st.columns([1.1,.9])
    with a:
        st.markdown('<div class="card">',unsafe_allow_html=True); panel('Product Opportunity by Segment','Average propensity of available products')
        pivot=recs.merge(seg_features[['customer_id','segment']],on='customer_id').groupby(['segment','product_name']).propensity.mean().reset_index()
        pivot=pivot[pivot.product_name.isin(['Working Capital Loan','Cash Management','Trade Finance','Virtual Account','Payroll Service'])]
        mat=pivot.pivot(index='segment',columns='product_name',values='propensity').fillna(0)
        fig=px.imshow(mat*100,text_auto='.0f',aspect='auto',color_continuous_scale=['#f3faf5','#b7eacb','#39b86f'])
        fig.update_traces(texttemplate='%{text}%',hovertemplate='%{y}<br>%{x}: %{z:.1f}%<extra></extra>');fig.update_xaxes(side='top');fig.update_coloraxes(showscale=False)
        plot_clean(fig,280);st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card">',unsafe_allow_html=True); panel('Product Opportunity Distribution','Average propensity across the customer base')
        avg=recs.groupby('product_name').propensity.mean().sort_values(ascending=False).head(7).reset_index()
        fig=px.bar(avg,x='propensity',y='product_name',orientation='h',text=avg.propensity.map(lambda x:f'{x:.0%}'),color='propensity',color_continuous_scale=['#d8eaff','#2f80ed'])
        fig.update_traces(textposition='outside');fig.update_coloraxes(showscale=False);fig.update_xaxes(tickformat='.0%',gridcolor='#edf1f5');fig.update_yaxes(title='')
        plot_clean(fig,280);st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>',unsafe_allow_html=True)
    st.markdown('<div class="card">',unsafe_allow_html=True); panel('Customer Opportunity List','Customers with the strongest current product opportunity')
    show=filtered.sort_values('propensity',ascending=False).head(8)[['customer_id','industry','region','segment','annual_revenue','product_count','product_name','propensity']].copy()
    show.columns=['Customer ID','Industry','Region','Segment','Revenue (Rp)','Products','Top Recommendation','Propensity'];show['Revenue (Rp)']=show['Revenue (Rp)'].map(money);show['Propensity']=show['Propensity'].map(lambda x:f'{x:.0%}')
    st.dataframe(show,use_container_width=True,hide_index=True,height=290,column_config={'Customer ID':st.column_config.TextColumn(width='small'),'Propensity':st.column_config.TextColumn(width='small')});st.markdown('</div>',unsafe_allow_html=True)

# -----------------------------
# Customer 360
# -----------------------------
if page=='Customer 360':
    panel('Customer 360','Detailed profile, network behavior and product recommendations')
    ids=filtered.customer_id.tolist() or customer_table.customer_id.tolist()
    cid=st.selectbox('Select customer',ids,index=0)
    cust=customer_table[customer_table.customer_id==cid].iloc[0]
    cp=customer_products[customer_products.customer_id==cid].merge(products,on='product_id',how='left')
    rr=recs[recs.customer_id==cid].sort_values('propensity',ascending=False).head(3)
    tx_out=transactions[transactions.sender_customer_id==cid]; tx_in=transactions[transactions.receiver_customer_id==cid]
    counterpart=pd.concat([tx_out.receiver_customer_id,tx_in.sender_customer_id]).nunique()
    c360=st.columns([1.2,1,1,1])
    with c360[0]: st.markdown(f'<div class="card"><div class="kpi"><div class="kpi-icon" style="background:#fff1df">🏢</div><div><div class="kpi-label">Customer</div><div class="kpi-value" style="font-size:1.15rem">{cid}</div><div class="kpi-note">{cust.industry}</div></div></div></div>',unsafe_allow_html=True)
    for col,label,val in [(c360[1],'Region',cust.region),(c360[2],'Annual Revenue',money(cust.annual_revenue)),(c360[3],'Segment',cust.segment)]:
        with col: st.markdown(f'<div class="card"><div class="info-title">{label}</div><div class="info-value" style="font-size:1rem">{val}</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>',unsafe_allow_html=True)
    l,r=st.columns([1.35,.65])
    with l:
        st.markdown('<div class="card">',unsafe_allow_html=True);panel('Basic Information')
        i1,i2,i3=st.columns(3)
        for col,label,val in [(i1,'Avg Balance',money(cust.avg_balance)),(i2,'Employees',f'{int(cust.employees):,}'),(i3,'Customer Since',str(2026-int(cust.customer_age_years)))]:
            with col: st.markdown(f'<div class="info-title">{label}</div><div class="info-value">{val}</div>',unsafe_allow_html=True)
        st.markdown('<hr>',unsafe_allow_html=True);panel('Transaction Behavior','Last 12 months')
        q1,q2,q3,q4=st.columns(4)
        vals=[('↗ Outgoing',money(tx_out.amount.sum())),('↘ Incoming',money(tx_in.amount.sum())),('⇄ Transactions',f'{len(tx_out)+len(tx_in):,}'),('♧ Counterparties',f'{counterpart:,}')]
        for col,(lab,val) in zip([q1,q2,q3,q4],vals):
            with col: st.markdown(f'<div class="card" style="box-shadow:none;background:#f8fafc"><div class="info-title">{lab}</div><div class="info-value">{val}</div></div>',unsafe_allow_html=True)
        st.markdown('<hr>',unsafe_allow_html=True);panel('Transaction Network','Top 10 counterparties')
        nodes=set([cid]); edges=[]
        txc=transactions[(transactions.sender_customer_id==cid)|(transactions.receiver_customer_id==cid)].copy().sort_values('amount',ascending=False).head(30)
        for row in txc.itertuples():
            other=row.receiver_customer_id if row.sender_customer_id==cid else row.sender_customer_id; nodes.add(other);edges.append((cid,other,float(row.amount)))
        sg=nx.Graph();sg.add_nodes_from(nodes)
        for a1,b1,w in edges:
            if sg.has_edge(a1,b1): sg[a1][b1]['weight']+=w
            else: sg.add_edge(a1,b1,weight=w)
        pos=nx.spring_layout(sg,seed=42,k=.9)
        ex=[]
        for a1,b1 in sg.edges(): ex += [go.Scatter(x=[pos[a1][0],pos[b1][0],None],y=[pos[a1][1],pos[b1][1],None],mode='lines',line=dict(width=1,color='#b7c4d2'),hoverinfo='none',showlegend=False)]
        nxs=list(sg.nodes()); xs=[pos[n][0] for n in nxs]; ys=[pos[n][1] for n in nxs]
        sizes=[22 if n==cid else 12 for n in nxs]; colors=['#f28c28' if n==cid else '#3f8eea' for n in nxs]
        ex.append(go.Scatter(x=xs,y=ys,mode='markers+text',text=[n if n==cid else '' for n in nxs],textposition='bottom center',marker=dict(size=sizes,color=colors,line=dict(width=1,color='#fff')),hovertext=nxs,hoverinfo='text',showlegend=False))
        fig=go.Figure(ex);fig.update_layout(height=330,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(visible=False),yaxis=dict(visible=False));st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False});st.markdown('</div>',unsafe_allow_html=True)
    with r:
        st.markdown('<div class="card">',unsafe_allow_html=True);panel('Existing Products')
        for _,p in cp.iterrows(): st.markdown(f'<span class="tag tag-blue">{p.product_name}</span>&nbsp;',unsafe_allow_html=True)
        st.markdown('<hr>',unsafe_allow_html=True);panel('Top Product Recommendations')
        for idx,row in enumerate(rr.itertuples(),1):
            pct=row.propensity
            st.markdown(f'<div class="rec"><div class="rec-head"><div class="rec-rank">{idx}</div><div><div class="rec-name">{row.product_name}</div><div class="rec-score">Propensity <b>{pct:.0%}</b></div></div></div><div class="bar"><div style="width:{pct*100:.0f}%"></div></div></div>',unsafe_allow_html=True)
        if len(rr):
            top=rr.iloc[0]
            st.markdown('<hr>',unsafe_allow_html=True);panel(f'Why {top.product_name}?')
            reasons=[('Transaction Amount',.32),('Network Counterparties',.21),('Annual Revenue',.18),('Out Degree',.12),('Product Count',.09)]
            for lab,v in reasons: st.markdown(f'<div style="display:flex;align-items:center;gap:.4rem;margin:.25rem 0"><div style="width:112px;font-size:.61rem;color:#68788d">{lab}</div><div style="flex:1;height:7px;background:#edf1f5;border-radius:9px;overflow:hidden"><div style="height:100%;width:{v*100:.0f}%;background:#4b9ae8"></div></div><div style="width:28px;font-size:.61rem;font-weight:800;color:#50627a">{v:.2f}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# -----------------------------
# Other pages
# -----------------------------
if page=='Customer Segmentation':
    panel('Customer Segmentation','Explore behavioral clusters and customer profiles')
    sp=seg_features.groupby('segment').agg(Customers=('customer_id','count'),Avg_Revenue=('annual_revenue','mean'),Avg_Tx=('total_outgoing_amount','mean'),Avg_Counterparty=('unique_outgoing_counterparties','mean'),Avg_Products=('product_count','mean')).reset_index()
    a,b=st.columns([1.1,.9])
    with a:
        fig=px.bar(sp,x='segment',y='Customers',color='segment',color_discrete_sequence=['#2f80ed','#f28c28','#39b86f','#8267e8']);fig.update_layout(showlegend=False);plot_clean(fig,320)
    with b:
        show=sp.copy();show['Avg_Revenue']=show['Avg_Revenue'].map(money);show['Avg_Tx']=show['Avg_Tx'].map(money);show['Avg_Counterparty']=show['Avg_Counterparty'].round(0);show['Avg_Products']=show['Avg_Products'].round(1);st.dataframe(show,use_container_width=True,hide_index=True,height=320)
    st.markdown('<div class="card">',unsafe_allow_html=True);panel('Customers by Segment');st.dataframe(filtered[['customer_id','industry','region','segment','annual_revenue','product_count']].sort_values('annual_revenue',ascending=False).head(50),use_container_width=True,hide_index=True,height=420);st.markdown('</div>',unsafe_allow_html=True)

if page=='Product Opportunity':
    panel('Product Opportunity','Identify cross-sell opportunities by customer and product')
    avg=recs.groupby('product_name').propensity.mean().sort_values(ascending=False).reset_index()
    fig=px.bar(avg,x='product_name',y='propensity',text=avg.propensity.map(lambda x:f'{x:.0%}'),color='propensity',color_continuous_scale=['#d9eaff','#2f80ed']);fig.update_traces(textposition='outside');fig.update_coloraxes(showscale=False);fig.update_yaxes(tickformat='.0%',gridcolor='#edf1f5');plot_clean(fig,360)
    show=recs.merge(seg_features[['customer_id','segment','industry','region']],on='customer_id').sort_values('propensity',ascending=False).head(100).copy();show['propensity']=show.propensity.map(lambda x:f'{x:.0%}');st.dataframe(show[['customer_id','industry','region','segment','product_name','propensity']],use_container_width=True,hide_index=True,height=520)

if page=='Network Analysis':
    panel('Network Analysis','Transaction ecosystem and connected business customers')
    deg=gf.sort_values('out_degree',ascending=False).head(25).merge(customers,on='customer_id')
    a,b=st.columns([1.1,.9])
    with a:
        fig=px.bar(deg.sort_values('out_degree'),x='out_degree',y='customer_id',orientation='h',text='out_degree');fig.update_traces(marker_color='#2f80ed');fig.update_xaxes(title='Outgoing counterparties');fig.update_yaxes(title='');plot_clean(fig,520)
    with b:
        topnodes=deg.customer_id.head(12).tolist();sg=G.subgraph(topnodes).to_undirected();pos=nx.spring_layout(sg,seed=42)
        traces=[]
        for a1,b1 in sg.edges(): traces.append(go.Scatter(x=[pos[a1][0],pos[b1][0],None],y=[pos[a1][1],pos[b1][1],None],mode='lines',line=dict(width=1,color='#c4ced9'),hoverinfo='none',showlegend=False))
        traces.append(go.Scatter(x=[pos[n][0] for n in sg.nodes()],y=[pos[n][1] for n in sg.nodes()],mode='markers+text',text=list(sg.nodes()),textposition='top center',marker=dict(size=15,color='#3f8eea',line=dict(width=1,color='#fff')),showlegend=False))
        fig=go.Figure(traces);fig.update_layout(height=520,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(visible=False),yaxis=dict(visible=False));st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
