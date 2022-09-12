import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.write("""
# App Cluster Pelanggan
Aplikasi ini membuat segmentasi **Pelanggan pada perusahaan penerbangan**!, 
[Link dokumentasi Machine Learning di Github](https://github.com/irvansikajudin/Data-Science-Projects-Based-On-Data-Science-Bootcamp/blob/master/flight%20company%20customer%20%20segmentation/Day_23_HW_Solution_Machine_Learning_(Unsupervised)%20(1).ipynb)
""")
# st.write('App ini dibuat untuk melihat segmentasi pada perushaan penerbangan, dan pada ini anda dapat melihat LRFMC menggunakan grafik')
st.write('Anda dapat memilih beberapa fitur di sidebar ya... :)')
st.write('Loading kali ini akan sangat lama, karena visualisasi segmentasi membutuhkan banyak komputasi, tunggu sampai selesai ya untuk informasi yang lebih detail terkait project ini')
st.write('---')

st.sidebar.info('### Pilih Parameter Input')
jumlah_cluster = st.sidebar.selectbox('Tentukan Jumlah Cluster :',(3,2,3,4,5,6,7,8,9,10))
top_ranking = st.sidebar.selectbox('Mau lihat Top Rangking ?',('Ya','Tidak'))
distribusi = st.sidebar.selectbox('Mau lihat Distribusi Fiturnya ?',('Ya','Tidak'))
cluster_pca = st.sidebar.selectbox('Mau lihat Clusternya (PCA) ?',('Ya','Tidak'))
distribusi_user = st.sidebar.selectbox('Mau lihat Distribusi Usernya ?',('Ya','Tidak'))
radar_chart = st.sidebar.selectbox('Mau liihat Radar Chart LRFMC ?',('Ya','Tidak'))

data = pd.read_csv('dataset/flight.csv')
data = data.drop('MEMBER_NO', axis=1)
numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
data_num = data.select_dtypes(include=numerics)
data_cat = data.select_dtypes(include=['object'])

WORK_CITY = data['WORK_CITY'].value_counts().reset_index()
WORK_CITY.columns = ['WORK_CITY', 'FREQ']
WORK_CITY['PERCENTAGE'] = round((WORK_CITY['FREQ']/WORK_CITY['FREQ'].sum())*100,2)
WORK_CITY = WORK_CITY[0:10]

WORK_PROVINCE = data['WORK_PROVINCE'].value_counts().reset_index()
WORK_PROVINCE.columns = ['WORK_PROVINCE', 'FREQ']
WORK_PROVINCE['PERCENTAGE'] = round((WORK_PROVINCE['FREQ']/WORK_PROVINCE['FREQ'].sum())*100,2)
WORK_PROVINCE = WORK_PROVINCE[0:10]
WORK_PROVINCE.head(3)

WORK_COUNTRY = data['WORK_COUNTRY'].value_counts().reset_index()
WORK_COUNTRY.columns = ['WORK_COUNTRY', 'FREQ']
WORK_COUNTRY['PERCENTAGE'] = round((WORK_COUNTRY['FREQ']/WORK_COUNTRY['FREQ'].sum())*100,2)
WORK_COUNTRY = WORK_COUNTRY[0:10]
WORK_COUNTRY.head(3)

GENDER = data['GENDER'].value_counts().reset_index()
GENDER.columns = ['GENDER', 'FREQ']
GENDER['PERCENTAGE'] = round((GENDER['FREQ']/GENDER['FREQ'].sum())*100,2)
GENDER = GENDER[0:10]
GENDER.head(3)

cat_feature = ['GENDER','WORK_CITY','WORK_PROVINCE','WORK_COUNTRY']

f,ax = plt.subplots(2,2,figsize=(18,15))

g = sns.barplot(x='GENDER',y='FREQ',data=GENDER, palette='husl', ax=ax[0,0])
ax[0,0].set_title('Gender')
ax[0,0].set_xlabel('Gender')
ax[0,0].set_ylabel('Frequency')

g = sns.barplot(x='WORK_CITY', y ='FREQ', data=WORK_CITY, ax=ax[0,1], palette=sns.cubehelix_palette(reverse=True, start=0, n_colors=10))
ax[0,1].set_title('Top 10 Work City')
ax[0,1].set_xlabel('Work City')
ax[0,1].set_ylabel('Frequency')

g = sns.barplot(x='WORK_PROVINCE', y ='FREQ', data=WORK_PROVINCE, ax=ax[1,0], palette=sns.cubehelix_palette(reverse=True, start=0, n_colors=10))
ax[1,0].set_title('Top 10 Work Province')
ax[1,0].set_xlabel('Work Province')
ax[1,0].set_ylabel('Frequency')

g = sns.barplot(x='WORK_COUNTRY', y ='FREQ', data=WORK_COUNTRY, ax=ax[1,1], palette=sns.cubehelix_palette(reverse=True, start=0, n_colors=10))
ax[1,1].set_title('Top 10 Work Country')
ax[1,1].set_xlabel('Work Country')
ax[1,1].set_ylabel('Frequency')

if top_ranking == 'Ya':
    st.header('Top Ranking')
    st.pyplot(f)
    st.write('---')


f,ax = plt.subplots(2,2,figsize=(18,15))

g = sns.distplot(data['AGE'], ax=ax[0,0])
ax[0,0].set_title('Age Distribution')

g = sns.distplot(data['FLIGHT_COUNT'], ax=ax[0,1], color='red')
ax[0,1].set_title('Flight Count Distribution')

g = sns.distplot(data['SEG_KM_SUM'], ax=ax[1,0], color='green')
ax[1,0].set_title('the cumulative total distance traveled')

g = sns.distplot(data['avg_discount'], ax=ax[1,1], color='black')
ax[1,1].set_title('Avg Discount Distribution')

if distribusi == 'Ya':
    st.header('Distribusi')
    st.pyplot(f)
    st.write('---')

corr_= data_num.corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr_, annot=True, fmt = ".2f", cmap = "BuPu")


# # Data Preprocessing
data_missing_value = data.isnull().sum().reset_index()
data_missing_value.columns = ['feature','missing_value']
data_missing_value['percentage'] = round((data_missing_value['missing_value']/len(data))*100,2)
data_missing_value = data_missing_value.sort_values('percentage', ascending=False).reset_index(drop=True)
data_missing_value = data_missing_value[data_missing_value['percentage']>0]


fig, ax = plt.subplots(figsize=(15,7))

g = sns.barplot(x = 'feature',y='percentage',data=data_missing_value,ax=ax, color='#00005f')

x = np.arange(len(data_missing_value['feature']))
y = data_missing_value['percentage']

for i, v in enumerate(y):
    ax.text(x[i]- 0.15, v+0.08, str(v)+'%', fontsize = 16, color='gray', fontweight='bold')
    
title = '''
Distribution of missing value
'''
ax.text(1.7,4.8,title,horizontalalignment='left',color='black',fontsize=22,fontweight='bold')
    

text = '''
Terdapat 6 fitur yang memiliki missing value (nilai yang hilang)
3 numerik dan 3 non numerik

WORK_PROVINCE       : 3.248 missing value (5.16%)
WORK_CITY                  : 2.269 missing value (3.60%)
SUM_YR_1                    : 551 missing value (0.87%)
AGE                                : 420 missing value (0.67%)
SUM_YR_2                    : 138 missing value (0.22%)
WORK_COUNTRY        : 26 missing value (0.04%)
'''
ax.text(1.7,2.5,text,horizontalalignment='left',color='black',fontsize=15,fontweight='normal')
    
ax.set_ylim(0,5.9)

ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

data = data.dropna()


# ## Feature Selection

# **berdasarkan referensi beberapa paper, bahwa kita bisa melakukan clustering dengan fitur RFM. Tetapi pada kasus ini, kita akan modifikasi RFM menjadi LRFMC**

# **Informasi tentang RFM**<br>
# * R – Recency – Keterkinian: Keterkinian pembelian adalah alat penting untuk mengidentifikasi pelanggan yang telah membeli sesuatu baru-baru ini. Pelanggan yang membeli belum lama ini lebih cenderung bereaksi terhadap penawaran baru daripada pelanggan yang pembeliannya terjadi sejak lama. Ini adalah faktor yang paling penting dalam analisis RFM.<br><br>
# * F – Frequency – Frekuensi: Frekuensi pembelian muncul setelah keterkinian. Jika pelanggan membeli lebih sering, kemungkinan respons positif lebih tinggi daripada pelanggan yang jarang membeli sesuatu.<br><br>
# * M – Monetary Value – Nilai Uang: Omset pembelian atau nilai moneter mengacu pada semua pembelian yang dilakukan oleh pelanggan. Pelanggan yang menghabiskan lebih banyak uang untuk pembelian lebih cenderung menanggapi penawaran daripada pelanggan yang telah menghabiskan jumlah yang lebih kecil
# 

# Link referensi paper RFM atau LRFM :
# * https://ieeexplore.ieee.org/document/8592638
# * https://ieeexplore.ieee.org/abstract/document/9085407
# * https://ieeexplore.ieee.org/document/7545236/
# * https://ieeexplore.ieee.org/document/7057094/similar
# 
# jika tidak punya akses download atau full read bisa akses di link google drive ini :)<br>
# https://drive.google.com/drive/folders/1R2zzE8cTf-JwNzP-orxaGDWoQGh2qmAy?usp=sharing

# **LRFMC pada Airline customer**<br>
# * L =  LOAD_TIME - FFP_DATE.<br>
# the number of months between the time of membership and the end of observation window = the end time of observation window - the time of membership<br><br>
# * R = LAST_TO_END<br>
# the number of months from the last time the customer took the company's aircraft to the end of the observation windows = the time from the last flight to the end of the observation window<br><br>
# * F = FLIGHT_COUNT<br>
# number of times the customer takes the company's aircraft in the observation window = number of flight in the observation window<br><br>
# * M = SEG_KM_SUM<br>
# Accumulated flight history of the customer in observation time = total flight kilometers of observation window<br><br>
# 
# * C = AVG_DISCOUNT
# average value of the discount coefficient corresponding to the passenger space during the observation time = average discount rate
# 

data = data[data['SUM_YR_1'].notnull()]
data = data[data['SUM_YR_2'].notnull()]
 
 # Only keep records where the fare is non-zero, or the average discount rate is 0 at the same time as the total number of kilometers traveled.
index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM']==0) & (data['avg_discount']==0)
data = data[index1 | index2 | index3]
 #Integrate data into the data variable


data = data[['FFP_DATE','LOAD_TIME', 'FLIGHT_COUNT', 'avg_discount', 'SEG_KM_SUM','LAST_TO_END']]

data['LOAD_TIME'] = pd.to_datetime(data['LOAD_TIME'])
data['FFP_DATE'] = pd.to_datetime(data['FFP_DATE'])
 
 
 # data_LRFMC data
data_LRFMC = pd.DataFrame()
# data_LRFMC.columns = ['L', 'R', 'F','M', 'C']
data_LRFMC['L'] =((data['LOAD_TIME'] - data['FFP_DATE']).dt.days/30)
data_LRFMC['R'] = data['LAST_TO_END']
data_LRFMC['F'] = data['FLIGHT_COUNT']
data_LRFMC['M'] = data['SEG_KM_SUM']
data_LRFMC['C'] = data['avg_discount']


# ## Duplicate Values
data_LRFMC.duplicated().sum()
data_LRFMC = data_LRFMC.drop_duplicates()


# ## Outliers
# ### Log Transformation
features = list(data_LRFMC)


# plt.figure(figsize=(10, 10))
# for i in range(0, len(features)):
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     plt.subplot(7, 1, i+1)
#     sns.boxplot(data_LRFMC[features[i]],orient='h',color='green')
#     # plt.tight_layout() 
#     # st.pyplot(bbox_inches='tight') # untuk streamlit

# plt.figure(figsize=(10, 10))
# for i in range(0, len(features)):
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     plt.subplot(7, 1, i+1)
#     sns.boxplot(np.log1p(data_LRFMC[features[i]])+1,orient='h',color='green')
#     # plt.tight_layout()
#     # st.pyplot(bbox_inches='tight') # kode untuk streamlit
    

data_LRFMC['L'] = np.log1p(data_LRFMC['L'])
data_LRFMC['R'] = np.log1p(data_LRFMC['R'])
data_LRFMC['F'] = np.log1p(data_LRFMC['F'])
data_LRFMC['M'] = np.log1p(data_LRFMC['M'])
data_LRFMC['C'] = np.log1p(data_LRFMC['C'])


# ### Remove outlier based on IQR
Q1 = data_LRFMC['C'].quantile(0.25)
Q3 = data_LRFMC['C'].quantile(0.75)
IQR = Q3 - Q1
low_limit = Q1 - (1.5 * IQR)
high_limit = Q3 + (1.5 * IQR)
filtered_entries = ((data_LRFMC['C'] >= low_limit) & (data_LRFMC['C'] <= high_limit))
data_LRFMC = data_LRFMC[filtered_entries]

Q1 = data_LRFMC['F'].quantile(0.25)
Q3 = data_LRFMC['F'].quantile(0.75)
IQR = Q3 - Q1
low_limit = Q1 - (1.5 * IQR)
high_limit = Q3 + (1.5 * IQR)
filtered_entries = ((data_LRFMC['F'] >= low_limit) & (data_LRFMC['F'] <= high_limit))
data_LRFMC = data_LRFMC[filtered_entries]

LRFMC = data_LRFMC

# plt.figure(figsize=(10, 10))
# for i in range(0, len(features)):
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     plt.subplot(7, 1, i+1)
#     sns.boxplot(data_LRFMC[features[i]],orient='h',color='green')
#     # plt.tight_layout() 
#     # st.pyplot(bbox_inches='tight') # kode untuk streamlit


# ## Scaling
data_LRFMC_std = StandardScaler().fit_transform(data_LRFMC)
scaled_data_LRFMC = pd.DataFrame(data_LRFMC_std, columns=list(data_LRFMC))
scaled_data_LRFMC.head(3)


# # Modeling
# ## Find the best K
arr_inertia = []
for i in range(2,9):
    kmeans = KMeans(n_clusters=i, random_state=31).fit(scaled_data_LRFMC)
    arr_inertia.append(kmeans.inertia_) # Sum of squared distances of samples to their closest cluster center.
    
fig, ax = plt.subplots(figsize=(15, 5))
sns.lineplot(x=range(2,9), y=arr_inertia, color='#000087', linewidth = 4)
sns.scatterplot(x=range(2,9), y=arr_inertia, s=300, color='#800000',  linestyle='--')


# ## Clustering
kmeans = KMeans(n_clusters=jumlah_cluster, random_state=7).fit(scaled_data_LRFMC)
scaled_data_LRFMC['cluster'] = kmeans.labels_
LRFMC['cluster'] = kmeans.labels_


# ## Visualiasasi clustering
if cluster_pca == 'Ya':
    pca = PCA(n_components=2)

    pca.fit(scaled_data_LRFMC)
    pcs = pca.transform(scaled_data_LRFMC)

    data_pca = pd.DataFrame(data = pcs, columns = ['PC 1', 'PC 2'])
    data_pca['clusters'] = kmeans.labels_


    fig, ax = plt.subplots(figsize=(15,10))
    if jumlah_cluster == 1:
        palette_color=['#000087']
    elif jumlah_cluster == 2:
        palette_color=['#000087','#800000']
    elif jumlah_cluster == 3:
        palette_color=['#000087','#800000','#005f00']
    elif jumlah_cluster == 4:
        palette_color=['#000087','#800000','#005f00',"#808000"]
    elif jumlah_cluster == 5:
        palette_color=['#000087','#800000','#005f00',"#808000",'#808080']
    elif jumlah_cluster == 6:
        palette_color=['#000087','#800000','#005f00',"#808000",'#808080','#EE6983']
    elif jumlah_cluster == 7:
        palette_color=['#000087','#800000','#005f00',"#808000",'#808080','#EE6983','#C3FF99']
    elif jumlah_cluster == 8:
        palette_color=['#000087','#800000','#005f00',"#808000",'#808080','#EE6983','#C3FF99','#F94892']
    elif jumlah_cluster == 9:
        palette_color=['#000087','#800000','#005f00',"#808000",'#808080','#EE6983','#C3FF99','#F94892','#FFC090']
    elif jumlah_cluster == 10:
        palette_color=['#000087','#800000','#005f00',"#808000",'#808080','#EE6983','#C3FF99','#F94892','#FFC090','#FF1E00']
    
    sns.scatterplot(
        x="PC 1", y="PC 2",
        hue="clusters",
        edgecolor='white',
        linestyle='--',
        data=data_pca,
        palette= palette_color,
        s=160,
        ax=ax
    )

    if cluster_pca == 'Ya':
        st.header('Visualisasi Cluster dengan PCA')
        st.pyplot(fig)
        st.write('---')
else:
    st.header('Visualisasi Cluster dengan PCA')
    st.warning('Anda sedang tidak menampilkan visualiasasi clustering dengan PCA, Jika anda ingin menampilkannya, aktifkan di sidebar, namun ingat proses komputasi akan menjadi sangat lama, sehingga anda membutuhkan waktu lebih untuk menuggu proses load visualisasi.')
    

# # Insight - Analysis Clustering
# **Re-transform numpy log**
LRFMC['L'] = np.expm1(LRFMC['L'])
LRFMC['R'] = np.expm1(LRFMC['R'])
LRFMC['F'] = np.expm1(LRFMC['F'])
LRFMC['M'] = np.expm1(LRFMC['M'])
LRFMC['C'] = np.expm1(LRFMC['C'])

def LClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
## for Recency 

def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
## for Frequency and Monetary value 

def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4    
    
## for Coeficient

def CClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4



quartiles = LRFMC.quantile(q=[0.25,0.50,0.75])
print(quartiles, type(quartiles))

quartiles=quartiles.to_dict()

LRFMC['L_Quartile'] = LRFMC['L'].apply(LClass, args=('L',quartiles,))
LRFMC['R_Quartile'] = LRFMC['R'].apply(RClass, args=('R',quartiles,))
LRFMC['F_Quartile'] = LRFMC['F'].apply(FMClass, args=('F',quartiles,))
LRFMC['M_Quartile'] = LRFMC['M'].apply(FMClass, args=('M',quartiles,))
LRFMC['C_Quartile'] = LRFMC['C'].apply(CClass, args=('C',quartiles,))

LRFMC['LRFMCClass'] = LRFMC.L_Quartile.map(str)                     + LRFMC.R_Quartile.map(str)                     + LRFMC.F_Quartile.map(str)                     + LRFMC.M_Quartile.map(str)                     + LRFMC.C_Quartile.map(str)

cluster_distribution = LRFMC['cluster'].value_counts().reset_index()
cluster_distribution.columns = ['cluster','number of users']

# cluster_distribution
# ## Cluster distribution
fig, ax = plt.subplots(figsize=(15,7))

g = sns.barplot(x = 'cluster',y='number of users',data=cluster_distribution,ax=ax)
ax.bar_label(ax.containers[0])

x = np.arange(len(cluster_distribution['cluster']))
y = cluster_distribution['number of users']


if distribusi_user == 'Ya':
    st.header('Total User pada setiap cluster')
    st.pyplot(fig)
    st.write('Total Semua User adalah : ',cluster_distribution['number of users'].sum())
    st.write('---')

# ## Cluster Characteristics
median_cluster = LRFMC.groupby('cluster')['L','R','F','M','C'].agg(['median']).reset_index()
mean_cluster = LRFMC.groupby('cluster')['L','R','F','M','C'].agg(['mean']).reset_index()
st.header('Median dan Mean')
median_cluster
mean_cluster
st.write('---')


median_cluster.columns = ['cluster', 'L','R','F','M','C']
median_cluster['L_Quartile'] = median_cluster['L'].apply(LClass, args=('L',quartiles,))
median_cluster['R_Quartile'] = median_cluster['R'].apply(RClass, args=('R',quartiles,))
median_cluster['F_Quartile'] = median_cluster['F'].apply(FMClass, args=('F',quartiles,))
median_cluster['M_Quartile'] = median_cluster['M'].apply(FMClass, args=('M',quartiles,))
median_cluster['C_Quartile'] = median_cluster['C'].apply(CClass, args=('C',quartiles,))

median_cluster['LRFMCClass'] = median_cluster.L_Quartile.map(str)                                 + median_cluster.R_Quartile.map(str)                                 + median_cluster.F_Quartile.map(str)                                 + median_cluster.M_Quartile.map(str)                                 + median_cluster.C_Quartile.map(str)

# median_cluster
# fig,ax = plt.subplots(5,1,figsize=(20,18))

# sns.barplot(x = 'cluster',y='L',data=median_cluster,ax=ax[0])
# sns.barplot(x = 'cluster',y='R',data=median_cluster,ax=ax[1])
# sns.barplot(x = 'cluster',y='F',data=median_cluster,ax=ax[2])
# sns.barplot(x = 'cluster',y='M',data=median_cluster,ax=ax[3])
# sns.barplot(x = 'cluster',y='C',data=median_cluster,ax=ax[4])

if radar_chart == 'Ya':
    # st.plotly_chart(fig)
    st.header('Radar Chart LRFMC')

    r1=pd.Series(kmeans.labels_).value_counts()
    r2=pd.DataFrame(kmeans.cluster_centers_)
    r4=r2.T
    r5=r4.max()
    r5=r5.max()
    r6=r4.min()
    r6=r6.min() 

    import plotly.graph_objects as go

    # categories = ['L','R','F','M','C']
    categories = ['C','L','R','F','M']
    fig = go.Figure()
    for i in range(0,jumlah_cluster):
        c = i
        fig.add_trace(go.Scatterpolar(
            r=[r4[i][0],r4[i][1],r4[i][2],r4[i][3],r4[i][4]],
            theta=categories,
            fill='toself',
            name='Cluster ' + str(c)
        ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        #   range=[-1.5, 1.5]
        range=[r6,r5]
        )),
    showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)



    for i in range(0,jumlah_cluster):
        fig = go.Figure()
        c = i
        fig.add_trace(go.Scatterpolar(
            r=[r4[i][0],r4[i][1],r4[i][2],r4[i][3],r4[i][4]],
            theta=categories,
            fill='toself',
            name='Cluster ' + str(c)
        ))

        fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[r6,r5]
            )),
        showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write('---')

