#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;font-size:200%;;">Anomaly Detection on Machine Failures || Industrial  Machine Anomaly Detection</h2>
# <h3  style="text-align:center;">Keywords : <span class="label label-success">IoT</span> <span class="label label-success">Anomaly Detection</span> <span class="label label-success">Model Interpretability</span> <span class="label label-success">Model Comparison</span></h3>

# # Table of Contents<a id='top'></a>
# >1. [Overview](#1.-Overview)  
# >    * [Project Detail](#Project-Detail)
# >    * [Goal of this notebook](#Goal-of-this-notebook)
# >1. [Import libraries](#2.-Import-libraries)
# >1. [Load the dataset](#3.-Load-the-dataset)
# >1. [Pre-processing](#4.-Pre-processing)
# >    * [Anomaly Points](#Anomaly-Points)
# >    * [Datetime Information](#Datetime-Information)
# >1. [EDA](#5.-EDA)  
# >    * [Basic Analysis](#Basic-Analysis)
# >    * [Time Series Analysis](#Time-Series-Analysis)
# >1. [Modeling](#6.-Modeling)
# >    * [Model1. Hotelling's T2](#Model1.-Hotelling's-T2)
# >    * [Model2. One-Class SVM](#Model2.-One\-Class-SVM)
# >    * [Model3. Isolation Forest](#Model3.-Isolation-Forest)
# >    * [Model4. LOF](#Model4.-LOF)
# >    * [Model5. ChangeFinder](#Model5.-ChangeFinder)
# >    * [Model6. Variance Based Method](#Model6.-Variance-Based-Method)
# >    * [Saving Machine Learning Model](#Saving-Machine-Learning-Model)
# >    * [Model Comparison](#Model-Comparison)
# >1. [Conclusion](#7.-Conclusion)
# >1. [References](#8.-References)

# # 1. Overview
# ## Project Detail
# >In this project, we use [NAB-dataset](https://www.kaggle.com/boltzmannbrain/nab), which is a novel benchmark for evaluating algorithms for anomaly detection in several fields.  
# >There are 58 timeseries data from various kind of sources.
# >* **Real data**
# >    * realAWSCloudwatch
# >    * realAdExchange
# >    * realKnownCause
# >    * realTraffic
# >    * realTweets
# >* **Artificial data**
# >    * artificialNoAnomaly
# >    * artificialWithAnomaly
# >
# >In these dataset above, I picked up and analyzed **'machine_temperature_system_failure'** from realKnownCause dataset based on my buissiness interests.  
# >This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).
# 
# ## Goal of this notebook
# >* Practice data pre-processing technique
# >* Practice EDA technique to deal with time-series data
# >* Practice visualising technique
# >* Practice anomaly detection modeling technique
# >    * from simple techniques to complex techniques
# >* Practice improving model interpretability technique
# >    * SHAP

# # 2. Import libraries

# In[1]:


# !pip install holoviews


# In[2]:


# !pip install changefinder


# In[3]:


# !pip install streamlit


# In[4]:


import streamlit as st


# In[5]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from matplotlib import pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import changefinder
from sklearn.metrics import f1_score
import shap
shap.initjs()
from tabulate import tabulate
from IPython.display import HTML, display


# # 3. Load the dataset
# >As above, we use 'machine_temperature_system_failure.csv' for our analysis.  
# >According to dataset information, it has the following features : 
# >* Temperature sensor data of an internal component of a large, industrial mahcine.
# >* The first anomaly is a planned shutdown of the machine. 
# >* The second anomaly is difficult to detect and directly led to the third anomaly, a catastrophic failure of the machine.

# In[6]:


# for dirname, _, filenames in os.walk('/kaggle/input/nab/realKnownCause/realKnownCause'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[7]:


# df = pd.read_csv("/kaggle/input/nab/realKnownCause/realKnownCause/machine_temperature_system_failure.csv",low_memory=False)
# print(f'machine_temperature_system_failure.csv : {df.shape}')
# df.head(3)


# In[72]:


# df = pd.read_csv('C:/Users/irvan/Downloads/machine_temperature_system_failure.csv')
df = pd.read_csv('dataset/machine_temperature_system_failure.csv')
df.head(3)


# # 4. Pre-processing

# ## Anomaly Points
# >We can get anomaly points information [here](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json)

# In[9]:


anomaly_points = [
        ["2013-12-10 06:25:00.000000","2013-12-12 05:35:00.000000"],
        ["2013-12-15 17:50:00.000000","2013-12-17 17:00:00.000000"],
        ["2014-01-27 14:20:00.000000","2014-01-29 13:30:00.000000"],
        ["2014-02-07 14:55:00.000000","2014-02-09 14:05:00.000000"]
]


# In[10]:


df['timestamp'] = pd.to_datetime(df['timestamp'])
#is anomaly? : True => 1, False => 0
df['anomaly'] = 0
for start, end in anomaly_points:
    df.loc[((df['timestamp'] >= start) & (df['timestamp'] <= end)), 'anomaly'] = 1


# ## Datetime Information

# In[11]:


df['year'] = df['timestamp'].apply(lambda x : x.year)
df['month'] = df['timestamp'].apply(lambda x : x.month)
df['day'] = df['timestamp'].apply(lambda x : x.day)
df['hour'] = df['timestamp'].apply(lambda x : x.hour)
df['minute'] = df['timestamp'].apply(lambda x : x.minute)


# In[12]:


df.index = df['timestamp']
df.drop(['timestamp'], axis=1, inplace=True)
df.head(3)


# In[13]:


# df = df.loc[(df['year']==2013) & (df['month']==5) & (df['day']==5),['value', 'anomaly', 'year', 'month', 'day', 'hour', 'minute']]


# In[14]:


# df5 = df.copy()


# In[15]:


# d5 = df5[(df5['year'] == 2014) & (df5['month'] == 2) & (df5['day'] == 2)][['value', 'anomaly', 'year', 'month', 'day', 'hour', 'minute']]


# In[16]:


# d5.head(3)


# # 5. EDA

# ## Basic Analysis

# In[17]:


count = hv.Bars(df.groupby(['year','month'])['value'].count()).opts(ylabel="Count", title='Year/Month Count')
mean = hv.Bars(df.groupby(['year','month']).agg({'value': ['mean']})['value']).opts(ylabel="Temperature", title='Year/Month Mean Temperature')
(count + mean).opts(opts.Bars(width=380, height=300,tools=['hover'],show_grid=True, stacked=True, legend_position='bottom'))


# In[18]:


year_maxmin = df.groupby(['year','month']).agg({'value': ['min', 'max']})
(hv.Bars(year_maxmin['value']['max']).opts(ylabel="Temperature", title='Year/Month Max Temperature') + hv.Bars(year_maxmin['value']['min']).opts(ylabel="Temperature", title='Year/Month Min Temperature'))    .opts(opts.Bars(width=380, height=300,tools=['hover'],show_grid=True, stacked=True, legend_position='bottom'))


# In[19]:


hv.Distribution(df['value']).opts(opts.Distribution(title="Temperature Distribution", xlabel="Temperature", ylabel="Density", width=700, height=300,tools=['hover'],show_grid=True))


# In[20]:


((hv.Distribution(df.loc[df['year']==2013,'value'], label='2013') * hv.Distribution(df.loc[df['year']==2014,'value'], label='2014')).opts(title="Temperature by Year Distribution", legend_position='bottom') + (hv.Distribution(df.loc[df['month']==12,'value'], label='12') * hv.Distribution(df.loc[df['month']==1,'value'], label='1')      * hv.Distribution(df.loc[df['month']==2,'value'], label='2')).opts(title="Temperature by Month Distribution", legend_position='bottom'))      .opts(opts.Distribution(xlabel="Temperature", ylabel="Density", width=380, height=300,tools=['hover'],show_grid=True))


# ## Time Series Analysis

# >plot temperature & its given anomaly points.

# In[21]:


anomalies = [[ind, value] for ind, value in zip(df[df['anomaly']==1].index, df.loc[df['anomaly']==1,'value'])]
(hv.Curve(df['value'], label="Temperature") * hv.Points(anomalies, label="Anomaly Points").opts(color='red', legend_position='bottom', size=2, title="Temperature & Given Anomaly Points"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))


# In[22]:


hv.Curve(df['value'].resample('D').mean()).opts(opts.Curve(title="Temperature Mean by Day", xlabel="Time", ylabel="Temperature", width=700, height=300,tools=['hover'],show_grid=True))


# # Deploy

# In[23]:


st.sidebar.header('Select The Date')


# In[24]:


df2 = df.copy()


# In[69]:


st.write("""
# Anomaly Detection on Machine Failures with Temperature
""")

st.markdown("""
<style>
.big-font {
    font-size:14px !important;
}
.bigg-font {
    font-size:14px !important;
    background-color:tomato !important;
    color:white !important;
}
.morebig-font {
    font-size:30px !important;
    text-align: center; 
    color: black;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="bigg-font">Created By : Irvan Sikajudin  || Linkedin : linkedin.com/in/irvansikajudin  || Github : github.com/irvansikajudin</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This Project Using 3 Models, Hotellings T2, Isolation Forest and Variance Based Method. </p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">on the top page, you will see a graph that is adjusted to the selected date, on the middle to the bottom page you can see a graph that shows the detection of anomalies in all existing data with the various methods provided. </p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

# st.markdown('<p class="big-font"></p>', unsafe_allow_html=True)




# In[26]:


# pip install --force-reinstall --no-deps bokeh==2.4.3


# In[27]:


# pip list


# In[28]:


def user_input_features():
    year = st.sidebar.slider('Year', 2013, 2014, 2013)
    month = st.sidebar.slider('Month', 1, 12, 12)
    day = st.sidebar.slider('Day', 1, 31, 16)
    data = {'year': year,
            'month': month,
            'day': day}
    features = pd.DataFrame(data, index=[0])
    return features


# In[29]:


df3 = user_input_features()

tgl1 = df3['month'][0],df3['day'][0],df3['year'][0]
tgl = str(tgl1)
tgl = 'Selected date for anomaly detection : ' + tgl
tgl = str(tgl)
st.info(tgl)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.write(""" Anomaly Detection Date : """ )

# with col2:
#     st.write(df3['month'][0], df3['day'][0], df3['year'][0])

# with col3:
#     st.write('')


# In[30]:


df4 = df.copy()


# In[31]:


df44 = df4[(df4['year'] == df3['year'][0]) & (df4['month'] == df3['month'][0]) & (df4['day'] == df3['day'][0])][['value', 'anomaly', 'year', 'month', 'day', 'hour', 'minute']]


# In[32]:


df44.head(4)


# ## Time Series Analysis

# In[33]:


anomalies = [[ind, value] for ind, value in zip(df44[df44['anomaly']==1].index, df44.loc[df44['anomaly']==1,'value'])]
aa = (hv.Curve(df44['value'], label="Temperature") * hv.Points(anomalies, label="Anomaly Points").opts(color='red', legend_position='bottom', size=2, title="Temperature & Given Anomaly Points - Time Series Analysis - Specific Date"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
aa


# In[34]:


if not df44.empty:
    aa = hv.render(aa, backend="bokeh")
    st.markdown('<p class="bigg-font">The graph below for the temperature plot & its given anomaly points,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)
    st.bokeh_chart(aa)
else:
    st.error("There is no data on the date you selected, so the graph for anomaly detection on the specific date is omitted.")
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


# ## Hotelling's T2

# In[35]:


# df4[(df4['year'] == 2014) & (df4['month'] == 2) & (df4['day'] == 31)][['value', 'anomaly', 'year', 'month', 'day', 'hour', 'minute']]


# In[36]:


# if not df44.empty:
#     print('a')


# In[37]:


if not df44.empty:
    st.write("""
    # In this session you will see the prediction results of each model in finding anomalies, according to the selected date. 
    """)
    hotelling_df = pd.DataFrame()
    hotelling_df['value'] = df44['value']
    mean = hotelling_df['value'].mean()
    std = hotelling_df['value'].std()
    hotelling_df['anomaly_score'] = [((x - mean)/std) ** 2 for x in hotelling_df['value']]
    hotelling_df['anomaly_threshold'] = stats.chi2.ppf(q=0.95, df=1) # df=1 berarti degree of freedomnya 1, karena hanya ada 1 variabel saja(valeu) 
    hotelling_df['anomaly']  = hotelling_df.apply(lambda x : 1 if x['anomaly_score'] > x['anomaly_threshold'] else 0, axis=1)
else:
    print("kosong")


# In[38]:


if not df44.empty:
    st.markdown('<p class="bigg-font">The graph below shows results of Hotelling"s T2 model in finding anomalies.,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

    b = (hv.Curve(hotelling_df['anomaly_score'], label='Anomaly Score') * hv.Curve(hotelling_df['anomaly_threshold'], label='Threshold').opts(color='red', line_dash="dotdash"))       .opts(title="Hotelling's T2 - Anomaly Score & Threshold", xlabel="Time", ylabel="Anomaly Score", legend_position='bottom').opts(opts.Curve(width=700, height=400, show_grid=True, tools=['hover']))
    b
else:
    print("kosong")


# In[39]:


if not df44.empty:
    hotelling_df44 = pd.DataFrame()
    hotelling_df44['value'] = df44['value']
    mean = hotelling_df44['value'].mean()
    stddf44 = hotelling_df44['value'].std()
    hotelling_df44['anomaly_score'] = [((x - mean)/stddf44) ** 2 for x in hotelling_df44['value']]
    hotelling_df44['anomaly_threshold'] = stats.chi2.ppf(q=0.95, df=1) # df disini adalah degree of fredom, karna hanya ada 1 variable maka valuenya adalah 1, jika 2 variabl adalah 2
    hotelling_df44['anomaly']  = hotelling_df44.apply(lambda x : 1 if x['anomaly_score'] > x['anomaly_threshold'] else 0, axis=1)
else:
    print("kosong")


# In[40]:


if not df44.empty:
    bb1 = (hv.Curve(hotelling_df44['anomaly_score'], label='Anomaly Score') * hv.Curve(hotelling_df44['anomaly_threshold'], label='Threshold').opts(color='red', line_dash="dotdash"))       .opts(title="Hotelling's T2 - Anomaly Score & Threshold", xlabel="Time", ylabel="Anomaly Score", legend_position='bottom').opts(opts.Curve(width=700, height=400, show_grid=True, tools=['hover']))
    bb1
else:
    print('kosong')


# In[41]:


if not df44.empty:
    anomalies = [[ind, value] for ind, value in zip(hotelling_df44[hotelling_df44['anomaly']==1].index, hotelling_df44.loc[hotelling_df44['anomaly']==1,'value'])]
    bb1 = (hv.Curve(hotelling_df44['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Hotelling's T2 - Detected Points - Specific Date"))        .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
    bb1
else:
    print('kosong')


# In[42]:


if not df44.empty:
    bb1 = hv.render(bb1, backend="bokeh")
    st.bokeh_chart(bb1)
else:
    print('kosong')
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the Hotellings T2 graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


# ## Isolation Forest
# ><div class="alert alert-info" role="alert">
# ><ul>
# ><li>Unsupervised tree-based anomaly detection method.</li>
# ></ul>
# ></div>

# In[43]:


if not df44.empty:
    iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700)
    iforest_ret = iforest_model.fit_predict(df44['value'].values.reshape(-1, 1))
    iforest_df44 = pd.DataFrame()
    iforest_df44['value'] = df44['value']
    iforest_df44['anomaly']  = [1 if i==-1 else 0 for i in iforest_ret]
else:
    print('kosong')


# In[44]:


if not df44.empty:
    anomalies = [[ind, value] for ind, value in zip(iforest_df44[iforest_df44['anomaly']==1].index, iforest_df44.loc[iforest_df44['anomaly']==1,'value'])]
    cc1 = (hv.Curve(iforest_df44['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Isolation Forest - Detected Points - Specific Date"))        .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
    cc1
else:
    print('kosong')


# In[45]:


# if not df44.empty:
#     sample_train = df44['value'].values[np.random.randint(0, len(df44['value']), (100))].reshape(-1, 1)
#     explainer = shap.TreeExplainer(model=iforest_model, feature_perturbation="interventional", data=sample_train)
#     shap_values = explainer.shap_values(X=sample_train)
#     shap.summary_plot(shap_values=shap_values, features=sample_train, feature_names=['value'], plot_type="violin")
# else:
#     print('kosong')


# In[46]:


# if not df44.empty:
#     iforest_f1 = f1_score(df44['anomaly'], iforest_df44['anomaly'])
#     print(f'Isolation Forest F1 Score : {iforest_f1}')
# else:
#     print('kosong')


# In[47]:


if not df44.empty:
    st.markdown('<p class="bigg-font">The graph below shows results of Isolation Forest model in finding anomalies.,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

    cc1 = hv.render(cc1, backend="bokeh")
    st.bokeh_chart(cc1)
else:
    print('kosong')
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the Hotellings T2 graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


# ## Variance Based Method
# ><div class="alert alert-info" role="alert">
# ><ul>
# ><li>This is variance based method with assumption of the normal distribution against the data.</li>
# ></ul>
# ></div>

# In[48]:


if not df44.empty:
    sigma_df44 = pd.DataFrame()
    sigma_df44['value'] = df44['value']
    std = sigma_df44['value'].std()
    sigma_df44['anomaly_threshold_3r'] = mean + 1.5*std
    sigma_df44['anomaly_threshold_3l'] = mean - 1.5*std
    sigma_df44['anomaly']  = sigma_df44.apply(lambda x : 1 if (x['value'] > x['anomaly_threshold_3r']) or (x['value'] < x['anomaly_threshold_3l']) else 0, axis=1)
else:
    print('kosong')


# In[49]:


if not df44.empty:
    anomalies = [[ind, value] for ind, value in zip(sigma_df44[sigma_df44['anomaly']==1].index, sigma_df44.loc[sigma_df44['anomaly']==1,'value'])]
    d1 = (hv.Curve(sigma_df44['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Variance Based Method - Detected Points - Specific Date "))        .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
    d1
else:
    print('kosong')


# In[50]:


# if not df44.empty:
#     sigma_f1 = f1_score(df44['anomaly'], sigma_df44['anomaly'])
#     print(f'Variance Based Method F1 Score : {sigma_f1}')
# else:
#     print('kosong')


# In[51]:


if not df44.empty:
    st.markdown('<p class="bigg-font">The graph below shows results of Variance Based Method model in finding anomalies.,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

    d1 = hv.render(d1, backend="bokeh")
    st.bokeh_chart(d1)
else:
    print('kosong')
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the Hotellings T2 graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


# ## Time Series Analysis - Full data record

# In[ ]:


st.write("""
# In this session you will see The graph below for the temperature plot & its given anomaly points,  in the graph below will display all data record. 
""")
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)


# In[52]:


anomalies = [[ind, value] for ind, value in zip(df[df['anomaly']==1].index, df.loc[df['anomaly']==1,'value'])]
a = (hv.Curve(df['value'], label="Temperature") * hv.Points(anomalies, label="Anomaly Points").opts(color='red', legend_position='bottom', size=2, title="Temperature & Given Anomaly Points - Time Series Analysis - Full Data Record"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
a


# In[53]:


a = hv.render(a, backend="bokeh")
st.bokeh_chart(a)


# In[ ]:


st.write("""
# In this session you will see the prediction results of each model in finding anomalies for all data record. 
""")
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)


# ## Hotelling's T2 - Full Record
# ><div class="alert alert-info" role="alert">
# ><ul>
# ><li>Basic anomaly detection method based on statustics.</li>
# ></ul>
# ></div>

# In[54]:


hotelling_df = pd.DataFrame()
hotelling_df['value'] = df['value']
mean = hotelling_df['value'].mean()
std = hotelling_df['value'].std()
hotelling_df['anomaly_score'] = [((x - mean)/std) ** 2 for x in hotelling_df['value']]
hotelling_df['anomaly_threshold'] = stats.chi2.ppf(q=0.95, df=1)
hotelling_df['anomaly']  = hotelling_df.apply(lambda x : 1 if x['anomaly_score'] > x['anomaly_threshold'] else 0, axis=1)


# In[55]:


b = (hv.Curve(hotelling_df['anomaly_score'], label='Anomaly Score') * hv.Curve(hotelling_df['anomaly_threshold'], label='Threshold').opts(color='red', line_dash="dotdash"))   .opts(title="Hotelling's T2 - Anomaly Score & Threshold - Detected Points - Full Data Record", xlabel="Time", ylabel="Anomaly Score", legend_position='bottom').opts(opts.Curve(width=700, height=400, show_grid=True, tools=['hover']))
b


# In[56]:


anomalies = [[ind, value] for ind, value in zip(hotelling_df[hotelling_df['anomaly']==1].index, hotelling_df.loc[hotelling_df['anomaly']==1,'value'])]
b1 = (hv.Curve(hotelling_df['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Hotelling's T2 - Detected Points - Full Data Record"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
b1


# In[57]:


st.markdown('<p class="bigg-font">The graph below shows results of Hotelling"s T2 model in finding anomalies.,  in the graph below will display all data record.</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

b = hv.render(b, backend="bokeh")
st.bokeh_chart(b)
b1 = hv.render(b1, backend="bokeh")
st.bokeh_chart(b1)


# In[58]:


hotelling_f1 = f1_score(df['anomaly'], hotelling_df['anomaly'])
print(f'Hotelling\'s T2 F1 Score : {hotelling_f1}')


# ## Isolation Forest - Full Record
# ><div class="alert alert-info" role="alert">
# ><ul>
# ><li>Unsupervised tree-based anomaly detection method.</li>
# ></ul>
# ></div>

# In[59]:


iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700)
iforest_ret = iforest_model.fit_predict(df['value'].values.reshape(-1, 1))
iforest_df = pd.DataFrame()
iforest_df['value'] = df['value']
iforest_df['anomaly']  = [1 if i==-1 else 0 for i in iforest_ret]


# In[60]:


anomalies = [[ind, value] for ind, value in zip(iforest_df[iforest_df['anomaly']==1].index, iforest_df.loc[iforest_df['anomaly']==1,'value'])]
c = (hv.Curve(iforest_df['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Isolation Forest - Detected Points - Full Data Record"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
c


# In[61]:


# sample_train = df['value'].values[np.random.randint(0, len(df['value']), (100))].reshape(-1, 1)
# explainer = shap.TreeExplainer(model=iforest_model, feature_perturbation="interventional", data=sample_train)
# shap_values = explainer.shap_values(X=sample_train)
# shap.summary_plot(shap_values=shap_values, features=sample_train, feature_names=['value'], plot_type="violin")


# In[62]:


st.markdown('<p class="bigg-font">The graph below shows results of Isolation Forest model in finding anomalies.,  in the graph below will display all data record.</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

c = hv.render(c, backend="bokeh")
st.bokeh_chart(c)


# In[63]:


iforest_f1 = f1_score(df['anomaly'], iforest_df['anomaly'])
print(f'Isolation Forest F1 Score : {iforest_f1}')


# ## Model6. Variance Based Method
# ><div class="alert alert-info" role="alert">
# ><ul>
# ><li>This is variance based method with assumption of the normal distribution against the data.</li>
# ></ul>
# ></div>

# In[64]:


sigma_df = pd.DataFrame()
sigma_df['value'] = df['value']
std = sigma_df['value'].std()
sigma_df['anomaly_threshold_3r'] = mean + 1.5*std
sigma_df['anomaly_threshold_3l'] = mean - 1.5*std
sigma_df['anomaly']  = sigma_df.apply(lambda x : 1 if (x['value'] > x['anomaly_threshold_3r']) or (x['value'] < x['anomaly_threshold_3l']) else 0, axis=1)


# In[65]:


anomalies = [[ind, value] for ind, value in zip(sigma_df[sigma_df['anomaly']==1].index, sigma_df.loc[sigma_df['anomaly']==1,'value'])]
d=(hv.Curve(sigma_df['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Variance Based Method - Detected Points"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
d


# In[66]:


st.markdown('<p class="bigg-font">The graph below shows results of Variance Based Method model in finding anomalies.,  in the graph below will display all data record.</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)


d = hv.render(d, backend="bokeh")
st.bokeh_chart(d)


# In[67]:


sigma_f1 = f1_score(df['anomaly'], sigma_df['anomaly'])
print(f'Variance Based Method F1 Score : {sigma_f1}')


# ## Model Compar

# In[68]:


st.markdown('<p class="bigg-font">The result of Evaluation of each Model</p>', unsafe_allow_html=True)


dffinal = HTML('<h3>Evaluation - F1 Score</h3>'+tabulate([['F1 Score', hotelling_f1, iforest_f1, sigma_f1]],                      ["", "Hotelling's T2",  'Isolation Forest',   'Variance Based Method'], tablefmt="html"))
st.write(dffinal)

