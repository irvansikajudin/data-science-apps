import streamlit as st
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

df = pd.read_csv('dataset/machine_temperature_system_failure.csv')
# df.head(3)
anomaly_points = [
        ["2013-12-10 06:25:00.000000","2013-12-12 05:35:00.000000"],
        ["2013-12-15 17:50:00.000000","2013-12-17 17:00:00.000000"],
        ["2014-01-27 14:20:00.000000","2014-01-29 13:30:00.000000"],
        ["2014-02-07 14:55:00.000000","2014-02-09 14:05:00.000000"]
]
df['timestamp'] = pd.to_datetime(df['timestamp'])
#is anomaly? : True => 1, False => 0
df['anomaly'] = 0
for start, end in anomaly_points:
    df.loc[((df['timestamp'] >= start) & (df['timestamp'] <= end)), 'anomaly'] = 1

df['year'] = df['timestamp'].apply(lambda x : x.year)
df['month'] = df['timestamp'].apply(lambda x : x.month)
df['day'] = df['timestamp'].apply(lambda x : x.day)
df['hour'] = df['timestamp'].apply(lambda x : x.hour)
df['minute'] = df['timestamp'].apply(lambda x : x.minute)

df.index = df['timestamp']
df.drop(['timestamp'], axis=1, inplace=True)
# df.head(3)

count = hv.Bars(df.groupby(['year','month'])['value'].count()).opts(ylabel="Count", title='Year/Month Count')
mean = hv.Bars(df.groupby(['year','month']).agg({'value': ['mean']})['value']).opts(ylabel="Temperature", title='Year/Month Mean Temperature')
(count + mean).opts(opts.Bars(width=380, height=300,tools=['hover'],show_grid=True, stacked=True, legend_position='bottom'))

year_maxmin = df.groupby(['year','month']).agg({'value': ['min', 'max']})
(hv.Bars(year_maxmin['value']['max']).opts(ylabel="Temperature", title='Year/Month Max Temperature') + hv.Bars(year_maxmin['value']['min']).opts(ylabel="Temperature", title='Year/Month Min Temperature'))    .opts(opts.Bars(width=380, height=300,tools=['hover'],show_grid=True, stacked=True, legend_position='bottom'))

hv.Distribution(df['value']).opts(opts.Distribution(title="Temperature Distribution", xlabel="Temperature", ylabel="Density", width=700, height=300,tools=['hover'],show_grid=True))

((hv.Distribution(df.loc[df['year']==2013,'value'], label='2013') * hv.Distribution(df.loc[df['year']==2014,'value'], label='2014')).opts(title="Temperature by Year Distribution", legend_position='bottom') + (hv.Distribution(df.loc[df['month']==12,'value'], label='12') * hv.Distribution(df.loc[df['month']==1,'value'], label='1')      * hv.Distribution(df.loc[df['month']==2,'value'], label='2')).opts(title="Temperature by Month Distribution", legend_position='bottom'))      .opts(opts.Distribution(xlabel="Temperature", ylabel="Density", width=380, height=300,tools=['hover'],show_grid=True))


# ## Time Series Analysis
# >plot temperature & its given anomaly points.

anomalies = [[ind, value] for ind, value in zip(df[df['anomaly']==1].index, df.loc[df['anomaly']==1,'value'])]
(hv.Curve(df['value'], label="Temperature") * hv.Points(anomalies, label="Anomaly Points").opts(color='red', legend_position='bottom', size=2, title="Temperature & Given Anomaly Points"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))

hv.Curve(df['value'].resample('D').mean()).opts(opts.Curve(title="Temperature Mean by Day", xlabel="Time", ylabel="Temperature", width=700, height=300,tools=['hover'],show_grid=True))


# # Deploy
st.sidebar.header('Select The Date')
df2 = df.copy()
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
st.write('[Machine Learning Documentation (Ipynb)](https://github.com/irvansikajudin/Anomaly-Detection/blob/main/myapp.ipynb)')
st.markdown('<p class="big-font">This Project Using 3 Models, Hotellings T2, Isolation Forest and Variance Based Method. </p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">on the top page, you will see a graph that is adjusted to the selected date, on the middle to the bottom page you can see a graph that shows the detection of anomalies in all existing data with the various methods provided. </p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

def user_input_features():
    year = st.sidebar.slider('Year', 2013, 2014, 2013)
    month = st.sidebar.slider('Month', 1, 12, 12)
    day = st.sidebar.slider('Day', 1, 31, 16)
    data = {'year': year,
            'month': month,
            'day': day}
    features = pd.DataFrame(data, index=[0])
    return features


df3 = user_input_features()

tgl1 = df3['month'][0],df3['day'][0],df3['year'][0]
tgl = str(tgl1)
tgl = 'Selected date for anomaly detection : ' + tgl
tgl = str(tgl)
st.info(tgl)

df4 = df.copy()

df44 = df4[(df4['year'] == df3['year'][0]) & (df4['month'] == df3['month'][0]) & (df4['day'] == df3['day'][0])][['value', 'anomaly', 'year', 'month', 'day', 'hour', 'minute']]
# df44.head(4)


# ## Time Series Analysis
anomalies = [[ind, value] for ind, value in zip(df44[df44['anomaly']==1].index, df44.loc[df44['anomaly']==1,'value'])]
aa = (hv.Curve(df44['value'], label="Temperature") * hv.Points(anomalies, label="Anomaly Points").opts(color='red', legend_position='bottom', size=2, title="Temperature & Given Anomaly Points - Time Series Analysis - Specific Date"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
# aa

if not df44.empty:
    aa = hv.render(aa, backend="bokeh")
    st.markdown('<p class="bigg-font">The graph below for the temperature plot & its given anomaly points,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)
    st.bokeh_chart(aa, use_container_width=True)
else:
    st.error("There is no data on the date you selected, so the graph for anomaly detection on the specific date is omitted.")
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


# ## Hotelling's T2

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

if not df44.empty:
    st.markdown('<p class="bigg-font">The graph below shows results of Hotelling"s T2 model in finding anomalies.,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

    b = (hv.Curve(hotelling_df['anomaly_score'], label='Anomaly Score') * hv.Curve(hotelling_df['anomaly_threshold'], label='Threshold').opts(color='red', line_dash="dotdash"))       .opts(title="Hotelling's T2 - Anomaly Score & Threshold", xlabel="Time", ylabel="Anomaly Score", legend_position='bottom').opts(opts.Curve(width=700, height=400, show_grid=True, tools=['hover']))
    # b
else:
    print("kosong")


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

if not df44.empty:
    bb1 = (hv.Curve(hotelling_df44['anomaly_score'], label='Anomaly Score') * hv.Curve(hotelling_df44['anomaly_threshold'], label='Threshold').opts(color='red', line_dash="dotdash"))       .opts(title="Hotelling's T2 - Anomaly Score & Threshold", xlabel="Time", ylabel="Anomaly Score", legend_position='bottom').opts(opts.Curve(width=700, height=400, show_grid=True, tools=['hover']))
    # bb1
else:
    print('kosong')

if not df44.empty:
    anomalies = [[ind, value] for ind, value in zip(hotelling_df44[hotelling_df44['anomaly']==1].index, hotelling_df44.loc[hotelling_df44['anomaly']==1,'value'])]
    bb1 = (hv.Curve(hotelling_df44['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Hotelling's T2 - Detected Points - Specific Date"))        .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
    # bb1
else:
    print('kosong')


if not df44.empty:
    bb1 = hv.render(bb1, backend="bokeh")
    st.bokeh_chart(bb1 , use_container_width=True)
else:
    print('kosong')
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the Hotellings T2 graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


if not df44.empty:
    iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700)
    iforest_ret = iforest_model.fit_predict(df44['value'].values.reshape(-1, 1))
    iforest_df44 = pd.DataFrame()
    iforest_df44['value'] = df44['value']
    iforest_df44['anomaly']  = [1 if i==-1 else 0 for i in iforest_ret]
else:
    print('kosong')



if not df44.empty:
    anomalies = [[ind, value] for ind, value in zip(iforest_df44[iforest_df44['anomaly']==1].index, iforest_df44.loc[iforest_df44['anomaly']==1,'value'])]
    cc1 = (hv.Curve(iforest_df44['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Isolation Forest - Detected Points - Specific Date"))        .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
    # cc1
else:
    print('kosong')


if not df44.empty:
    st.markdown('<p class="bigg-font">The graph below shows results of Isolation Forest model in finding anomalies.,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

    cc1 = hv.render(cc1, backend="bokeh")
    st.bokeh_chart(cc1, use_container_width=True)
else:
    print('kosong')
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the Hotellings T2 graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


if not df44.empty:
    sigma_df44 = pd.DataFrame()
    sigma_df44['value'] = df44['value']
    std = sigma_df44['value'].std()
    sigma_df44['anomaly_threshold_3r'] = mean + 1.5*std
    sigma_df44['anomaly_threshold_3l'] = mean - 1.5*std
    sigma_df44['anomaly']  = sigma_df44.apply(lambda x : 1 if (x['value'] > x['anomaly_threshold_3r']) or (x['value'] < x['anomaly_threshold_3l']) else 0, axis=1)
else:
    print('kosong')

if not df44.empty:
    anomalies = [[ind, value] for ind, value in zip(sigma_df44[sigma_df44['anomaly']==1].index, sigma_df44.loc[sigma_df44['anomaly']==1,'value'])]
    d1 = (hv.Curve(sigma_df44['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Variance Based Method - Detected Points - Specific Date "))        .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
    # d1
else:
    print('kosong')

if not df44.empty:
    st.markdown('<p class="bigg-font">The graph below shows results of Variance Based Method model in finding anomalies.,  in the graph below will only display data according to the selected date.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

    d1 = hv.render(d1, backend="bokeh")
    st.bokeh_chart(d1, use_container_width=True)
else:
    print('kosong')
#     st.markdown('<p class="big-font">There is no data on the date you selected, so the Hotellings T2 graph for anomaly detection on the specific date is omitted. </p>', unsafe_allow_html=True)


# ## Time Series Analysis - Full data record
st.write("""
# In this session you will see The graph below for the temperature plot & its given anomaly points,  in the graph below will display all data record. 
""")
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

anomalies = [[ind, value] for ind, value in zip(df[df['anomaly']==1].index, df.loc[df['anomaly']==1,'value'])]
a = (hv.Curve(df['value'], label="Temperature") * hv.Points(anomalies, label="Anomaly Points").opts(color='red', legend_position='bottom', size=2, title="Temperature & Given Anomaly Points - Time Series Analysis - Full Data Record"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
a = hv.render(a, backend="bokeh")
st.bokeh_chart(a, use_container_width=True)

st.write("""
# In this session you will see the prediction results of each model in finding anomalies for all data record. 
""")
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

hotelling_df = pd.DataFrame()
hotelling_df['value'] = df['value']
mean = hotelling_df['value'].mean()
std = hotelling_df['value'].std()
hotelling_df['anomaly_score'] = [((x - mean)/std) ** 2 for x in hotelling_df['value']]
hotelling_df['anomaly_threshold'] = stats.chi2.ppf(q=0.95, df=1)
hotelling_df['anomaly']  = hotelling_df.apply(lambda x : 1 if x['anomaly_score'] > x['anomaly_threshold'] else 0, axis=1)

b = (hv.Curve(hotelling_df['anomaly_score'], label='Anomaly Score') * hv.Curve(hotelling_df['anomaly_threshold'], label='Threshold').opts(color='red', line_dash="dotdash")) \
  .opts(title="Hotelling's T2 - Anomaly Score & Threshold - Detected Points - Full Data Record", xlabel="Time", ylabel="Anomaly Score", legend_position='bottom').opts(opts.Curve(width=700, height=400, show_grid=True, tools=['hover']))
# b

anomalies = [[ind, value] for ind, value in zip(hotelling_df[hotelling_df['anomaly']==1].index, hotelling_df.loc[hotelling_df['anomaly']==1,'value'])]
b1 = (hv.Curve(hotelling_df['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Hotelling's T2 - Detected Points - Full Data Record"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
# b1

st.markdown('<p class="bigg-font">The graph below shows results of Hotelling"s T2 model in finding anomalies.,  in the graph below will display all data record.</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

b = hv.render(b, backend="bokeh")
st.bokeh_chart(b, use_container_width=True)

b1 = hv.render(b1, backend="bokeh")
st.bokeh_chart(b1, use_container_width=True)


hotelling_f1 = f1_score(df['anomaly'], hotelling_df['anomaly'])
print(f'Hotelling\'s T2 F1 Score : {hotelling_f1}')

iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700)
iforest_ret = iforest_model.fit_predict(df['value'].values.reshape(-1, 1))
iforest_df = pd.DataFrame()
iforest_df['value'] = df['value']
iforest_df['anomaly']  = [1 if i==-1 else 0 for i in iforest_ret]

anomalies = [[ind, value] for ind, value in zip(iforest_df[iforest_df['anomaly']==1].index, iforest_df.loc[iforest_df['anomaly']==1,'value'])]
c = (hv.Curve(iforest_df['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Isolation Forest - Detected Points - Full Data Record"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
# c

st.markdown('<p class="bigg-font">The graph below shows results of Isolation Forest model in finding anomalies.,  in the graph below will display all data record.</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)

c = hv.render(c, backend="bokeh")
st.bokeh_chart(c, use_container_width=True)

iforest_f1 = f1_score(df['anomaly'], iforest_df['anomaly'])
print(f'Isolation Forest F1 Score : {iforest_f1}')

sigma_df = pd.DataFrame()
sigma_df['value'] = df['value']
std = sigma_df['value'].std()
sigma_df['anomaly_threshold_3r'] = mean + 1.5*std
sigma_df['anomaly_threshold_3l'] = mean - 1.5*std
sigma_df['anomaly']  = sigma_df.apply(lambda x : 1 if (x['value'] > x['anomaly_threshold_3r']) or (x['value'] < x['anomaly_threshold_3l']) else 0, axis=1)

anomalies = [[ind, value] for ind, value in zip(sigma_df[sigma_df['anomaly']==1].index, sigma_df.loc[sigma_df['anomaly']==1,'value'])]
d=(hv.Curve(sigma_df['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Variance Based Method - Detected Points"))    .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))
# d

st.markdown('<p class="bigg-font">The graph below shows results of Variance Based Method model in finding anomalies.,  in the graph below will display all data record.</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">This dataset does not include acutual anomaly point, so we need to refer to the [NAB github page](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json).</p>', unsafe_allow_html=True)


d = hv.render(d, backend="bokeh")
st.bokeh_chart(d, use_container_width=True)

sigma_f1 = f1_score(df['anomaly'], sigma_df['anomaly'])
print(f'Variance Based Method F1 Score : {sigma_f1}')

st.markdown('<p class="bigg-font">The result of Evaluation of each Model</p>', unsafe_allow_html=True)


dffinal = HTML('<h3>Evaluation - F1 Score</h3>'+tabulate([['F1 Score', hotelling_f1, iforest_f1, sigma_f1]],                      ["", "Hotelling's T2",  'Isolation Forest',   'Variance Based Method'], tablefmt="html"))
st.write(dffinal)



