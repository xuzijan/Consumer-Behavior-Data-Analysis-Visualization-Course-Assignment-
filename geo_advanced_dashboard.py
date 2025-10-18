"""
高级地理可视化分析仪表盘
- 动态地理热力图（随季节/时间变化）
- 地理聚类分布图
- 地区钻取与画像弹窗
- 地理网络流向图
- 多维雷达对比
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from sklearn.cluster import KMeans
import yaml

# 读取数据
df = pd.read_csv('shopping_behavior_updated.csv')

st.set_page_config(page_title="高级地理分析", layout="wide")
st.title("购物行为高级地理可视化分析")

# 侧边栏筛选
gender = st.sidebar.multiselect('性别', options=df['Gender'].unique(), default=list(df['Gender'].unique()))
category = st.sidebar.multiselect('商品类别', options=df['Category'].unique(), default=list(df['Category'].unique()))
season = st.sidebar.multiselect('季节', options=df['Season'].unique(), default=list(df['Season'].unique()))

# 数据筛选
df_filtered = df[
    (df['Gender'].isin(gender)) &
    (df['Category'].isin(category)) &
    (df['Season'].isin(season))
]

# 州名与缩写映射
def get_state_abbr():
    return {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
        'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
        'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

state_abbr = get_state_abbr()
df_filtered['StateAbbr'] = df_filtered['Location'].map(state_abbr)

# 动态地理热力图（按季节切换）
st.subheader('动态地理热力图（季节分布）')
season_selected = st.selectbox('选择季节', options=['全部'] + list(df['Season'].unique()))
if season_selected != '全部':
    df_season = df_filtered[df_filtered['Season'] == season_selected]
else:
    df_season = df_filtered
state_sales = df_season['Location'].value_counts().reset_index()
state_sales.columns = ['State', 'Sales']
state_sales['StateAbbr'] = state_sales['State'].map(state_abbr)
fig_map = px.choropleth(state_sales,
                        locations='StateAbbr',
                        locationmode='USA-states',
                        color='Sales',
                        color_continuous_scale='Reds',
                        scope='usa',
                        title=f'各州销量分布（{season_selected}）')
st.plotly_chart(fig_map, use_container_width=True)

# 地理聚类分布图
st.subheader('地区客户聚类分布')
cluster_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases']
X = df_filtered[cluster_cols].dropna()
if len(X) > 10:
    k = st.slider('聚类数K', 2, 8, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    df_filtered['Cluster'] = -1
    df_filtered.loc[X.index, 'Cluster'] = kmeans.labels_
    cluster_state = df_filtered.groupby(['Location', 'Cluster']).size().reset_index(name='Count')
    cluster_state['StateAbbr'] = cluster_state['Location'].map(state_abbr)
    fig_cluster_map = px.scatter_geo(cluster_state,
                                     locations='StateAbbr',
                                     locationmode='USA-states',
                                     color='Cluster',
                                     size='Count',
                                     scope='usa',
                                     title='各州客户聚类分布')
    st.plotly_chart(fig_cluster_map, use_container_width=True)
else:
    st.info('数据量不足，无法聚类')

# 地区钻取与画像弹窗
st.subheader('地区画像钻取')
state_pick = st.selectbox('选择州查看画像', options=list(df_filtered['Location'].unique()))
df_state = df_filtered[df_filtered['Location'] == state_pick]
if not df_state.empty:
    st.markdown(f"**{state_pick}画像：**")
    st.write(df_state.describe(include='all'))
else:
    st.info('该州暂无数据')

# 地理网络流向图（商品推荐/客户迁移）
st.subheader('地区商品推荐网络流向')
import networkx as nx
from itertools import combinations
item_pairs = df_filtered.groupby('Location')['Item Purchased'].apply(lambda x: list(set(x))).reset_index()
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

# 多维雷达对比
st.subheader('各州多维指标雷达图')
radar_cols = ['Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
radar_data = df_filtered.groupby('Location')[radar_cols].mean().reset_index()
if not radar_data.empty:
    for i in range(min(5, len(radar_data))):
        row = radar_data.iloc[i]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[row[c] for c in radar_cols],
                                            theta=radar_cols,
                                            fill='toself',
                                            name=row['Location']))
        fig_radar.update_layout(title=f'{row["Location"]}多维指标雷达图', polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info('暂无雷达数据')

st.markdown('---')
st.markdown('项目亮点：动态地理热力图、聚类分布、地区画像钻取、网络流向、雷达对比')
