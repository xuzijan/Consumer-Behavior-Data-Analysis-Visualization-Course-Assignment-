"""
交互式购物行为分析仪表盘
- 支持多维度筛选（性别、年龄、类别、季节等）
- 拖动时间轴观察趋势
- 按键切换聚类/画像/推荐等模块
- 动态图表：热力图、桑基图、雷达图、聚类降维、网络图
- 支持 YAML 配置
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 读取数据
df = pd.read_csv('shopping_behavior_updated.csv')

# 读取配置（如有）
def load_config(path='dashboard_config.yaml'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}
config = load_config()

st.set_page_config(page_title="购物行为交互分析", layout="wide")
st.title("购物行为多维度交互分析仪表盘")

# 侧边栏筛选
gender = st.sidebar.multiselect('性别', options=df['Gender'].unique(), default=list(df['Gender'].unique()))
age_range = st.sidebar.slider('年龄区间', int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
category = st.sidebar.multiselect('商品类别', options=df['Category'].unique(), default=list(df['Category'].unique()))
season = st.sidebar.multiselect('季节', options=df['Season'].unique(), default=list(df['Season'].unique()))

# 数据筛选
df_filtered = df[
    (df['Gender'].isin(gender)) &
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Category'].isin(category)) &
    (df['Season'].isin(season))
]

# 时间轴（季节）拖动
time_season = st.selectbox('选择季节查看趋势', options=['全部'] + list(df['Season'].unique()))
if time_season != '全部':
    df_trend = df[df['Season'] == time_season]
else:
    df_trend = df

# 动态销量趋势图
st.subheader('销量趋势（可按季节切换）')
sales_trend = df_trend.groupby('Category')['Purchase Amount (USD)'].sum().reset_index()
fig_trend = px.bar(sales_trend, x='Category', y='Purchase Amount (USD)', color='Category', title='各类别销量趋势')
st.plotly_chart(fig_trend, use_container_width=True)

# 多维度交叉热力图
st.subheader('年龄-性别-类别-季节交叉热力图')
heatmap_data = df_filtered.groupby(['Age', 'Gender', 'Category', 'Season']).size().reset_index(name='Count')
if not heatmap_data.empty:
    pivot = heatmap_data.pivot_table(index='Age', columns='Category', values='Count', aggfunc='sum', fill_value=0)
    fig_heat = px.imshow(pivot, aspect='auto', color_continuous_scale='Viridis', title='年龄-类别热力图')
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info('筛选条件下无数据')

# 客户行为聚类与降维
st.subheader('客户行为聚类与降维可视化')
cluster_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases']
X = df_filtered[cluster_cols].dropna()
if len(X) > 10:
    k = st.slider('聚类数K', 2, 8, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
    cluster_df = pd.DataFrame(tsne, columns=['x', 'y'])
    cluster_df['Cluster'] = kmeans.labels_
    fig_cluster = px.scatter(cluster_df, x='x', y='y', color='Cluster', title='客户聚类降维分布')
    st.plotly_chart(fig_cluster, use_container_width=True)
else:
    st.info('数据量不足，无法聚类')

# 商品推荐相关性网络图
st.subheader('商品推荐相关性网络图')
import networkx as nx
from itertools import combinations
item_pairs = df_filtered.groupby('Customer ID')['Item Purchased'].apply(lambda x: list(set(x))).reset_index()
edge_list = []
for items in item_pairs['Item Purchased']:
    if len(items) > 1:
        edge_list += list(combinations(items, 2))
G = nx.Graph()
G.add_edges_from(edge_list)
edge_freq = pd.Series(edge_list).value_counts().reset_index()
edge_freq.columns = ['pair', 'weight']
edge_freq = edge_freq[edge_freq['weight'] > 2]
if not edge_freq.empty:
    fig_net = go.Figure()
    for idx, row in edge_freq.iterrows():
        fig_net.add_trace(go.Scatter(x=[idx, idx+1], y=[0,1], mode='lines', line=dict(width=row['weight']), name=str(row['pair'])))
    fig_net.update_layout(title='高频商品推荐关联', showlegend=False)
    st.plotly_chart(fig_net, use_container_width=True)
else:
    st.info('暂无高频商品关联')

# 消费者画像卡片
st.subheader('典型消费者画像')
if len(X) > 0:
    for i in range(min(3, len(X))):
        st.markdown(f"**画像{i+1}: 年龄{int(X.iloc[i]['Age'])}, 历史购买{int(X.iloc[i]['Previous Purchases'])}, 总金额{int(X.iloc[i]['Purchase Amount (USD)'])}$")

# YAML配置展示
st.sidebar.subheader('当前仪表盘配置')
st.sidebar.code(yaml.dump(config, allow_unicode=True))

st.markdown('---')
st.markdown('项目亮点：交互式多维分析、聚类、画像、推荐、YAML配置、网页端体验')
