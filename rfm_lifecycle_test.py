"""
客户生命周期与流失预测（RFM模型+流失预警）
- RFM分层客户价值
- 流失概率预测
- 交互式可视化
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 读取数据
df = pd.read_csv('shopping_behavior_updated.csv')

st.set_page_config(page_title="客户生命周期与流失预测", layout="wide")
st.title("客户生命周期分层与流失预警分析")

# RFM计算
rfm = df.groupby('Customer ID').agg({
    'Previous Purchases': 'max',
    'Purchase Amount (USD)': 'sum',
    'Age': 'max'
}).reset_index()
rfm['Recency'] = np.random.randint(1, 12, size=len(rfm))  # 模拟最近一次购买距离当前的月数
rfm['Frequency'] = rfm['Previous Purchases']
rfm['Monetary'] = rfm['Purchase Amount (USD)']

# RFM分层
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].astype(int).sum(axis=1)

# RFM分布可视化
st.subheader('RFM客户分层分布')
fig_rfm = px.scatter(rfm, x='Frequency', y='Monetary', color='RFM_Score', size='Recency', hover_data=['Customer ID','RFM_Segment'])
st.plotly_chart(fig_rfm, use_container_width=True)

# 流失标签（模拟：RFM分数低于6视为流失风险）
rfm['Churn'] = (rfm['RFM_Score'] < 6).astype(int)

# 流失预测模型
st.subheader('流失概率预测')
X = rfm[['Recency','Frequency','Monetary']]
y = rfm['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
st.write(f'流失预测AUC：{auc:.2f}')
rfm_test = rfm.iloc[X_test.index].copy()
rfm_test['Churn_Prob'] = proba
fig_churn = px.histogram(rfm_test, x='Churn_Prob', nbins=20, color='Churn', title='客户流失概率分布')
st.plotly_chart(fig_churn, use_container_width=True)

# 高风险客户展示
st.subheader('高风险客户列表')
high_risk = rfm_test[rfm_test['Churn_Prob'] > 0.7]
st.dataframe(high_risk[['Customer ID','Churn_Prob','RFM_Segment','RFM_Score']])

st.markdown('---')
st.markdown('项目亮点：RFM分层、流失预测、交互式可视化')
