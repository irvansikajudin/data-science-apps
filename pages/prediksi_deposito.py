import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.write("""
# App Prediksi Deposito
Aplikasi ini memprediksi **Kemungkinan Pelanggan Melakukan Deposito**!, 
[Link dokumentasi Machine Learning di Github](https://github.com/irvansikajudin/Term-Deposit-Prediction-Bank-Marketing-Campaign-)
""")

dataset = pd.read_csv('dataset/deposito_clean.csv')
# dataset = pd.read_csv('https://drive.google.com/uc?export=download&id=1xFtcJgieQsq7jgY7F32Jo6vRHR3FQeMv')
X = dataset.drop('deposit', axis=1)
y = dataset['deposit']

# karena data terlalu besar maka akan saya split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = 0.2,
                                                random_state = 42)
# Sidebar
# Header of Specify Input Parameters
st.sidebar.info('### Pilih Parameter Input')
tipekarakteristikdeposito = st.sidebar.selectbox('Mau liat karakteristik pelanggan yg cendrung deposito?',('Ya','Tidak'))

def user_input_features():
    if tipekarakteristikdeposito == 'Ya':
        age=st.sidebar.slider('age',18,95,95)
        marital=st.sidebar.slider('marital',0,2,2)
        default=st.sidebar.slider('default',0,1,1)
        balance=st.sidebar.slider('balance',-6847,81204,81204)
        housing=st.sidebar.slider('housing',0,1,1)
        loan=st.sidebar.slider('loan',0,1,1)
        day=st.sidebar.slider('day',1,31,31)
        duration=st.sidebar.slider('duration',2,3881,985)
        campaign=st.sidebar.slider('campaign',1,63,63)
        pdays=st.sidebar.slider('pdays',-1,854,854)
        previous=st.sidebar.slider('previous',0,58,58)
        job_admin=st.sidebar.slider('job_admin',0,1,1)
        job_blue_collar=st.sidebar.slider('job_blue_collar',0,1,1)
        job_entrepreneur=st.sidebar.slider('job_entrepreneur',0,1,1)
        job_housemaid=st.sidebar.slider('job_housemaid',0,1,1)
        job_management=st.sidebar.slider('job_management',0,1,1)
        job_retired=st.sidebar.slider('job_retired',0,1,1)
        job_self_employed=st.sidebar.slider('job_self_employed',0,1,1)
        job_services=st.sidebar.slider('job_services',0,1,1)
        job_student=st.sidebar.slider('job_student',0,1,1)
        job_technician=st.sidebar.slider('job_technician',0,1,1)
        job_unemployed=st.sidebar.slider('job_unemployed',0,1,1)
        job_unknown=st.sidebar.slider('job_unknown',0,1,1)
        education_primary=st.sidebar.slider('education_primary',0,1,1)
        education_secondary=st.sidebar.slider('education_secondary',0,1,1)
        education_tertiary=st.sidebar.slider('education_tertiary',0,1,1)
        education_unknown=st.sidebar.slider('education_unknown',0,1,1)
        contact_cellular=st.sidebar.slider('contact_cellular',0,1,1)
        contact_telephone=st.sidebar.slider('contact_telephone',0,1,1)
        contact_unknown=st.sidebar.slider('contact_unknown',0,1,1)
        month_apr=st.sidebar.slider('month_apr',0,1,1)
        month_aug=st.sidebar.slider('month_aug',0,1,1)
        month_dec=st.sidebar.slider('month_dec',0,1,1)
        month_feb=st.sidebar.slider('month_feb',0,1,1)
        month_jan=st.sidebar.slider('month_jan',0,1,1)
        month_jul=st.sidebar.slider('month_jul',0,1,1)
        month_jun=st.sidebar.slider('month_jun',0,1,1)
        month_mar=st.sidebar.slider('month_mar',0,1,1)
        month_may=st.sidebar.slider('month_may',0,1,1)
        month_nov=st.sidebar.slider('month_nov',0,1,1)
        month_oct=st.sidebar.slider('month_oct',0,1,1)
        month_sep=st.sidebar.slider('month_sep',0,1,1)
        poutcome_failure=st.sidebar.slider('poutcome_failure',0,1,1)
        poutcome_other=st.sidebar.slider('poutcome_other',0,1,1)
        poutcome_success=st.sidebar.slider('poutcome_success',0,1,1)
        poutcome_unknown=st.sidebar.slider('poutcome_unknown',0,1,1)
    else:
        age=st.sidebar.slider('age',18,95,95)
        marital=st.sidebar.slider('marital',0,2,2)
        default=st.sidebar.slider('default',0,1,1)
        balance=st.sidebar.slider('balance',-6847,81204,81204)
        housing=st.sidebar.slider('housing',0,1,1)
        loan=st.sidebar.slider('loan',0,1,1)
        day=st.sidebar.slider('day',1,31,31)
        duration=st.sidebar.slider('duration',2,3881,75)
        campaign=st.sidebar.slider('campaign',1,63,63)
        pdays=st.sidebar.slider('pdays',-1,854,854)
        previous=st.sidebar.slider('previous',0,58,58)
        job_admin=st.sidebar.slider('job_admin',0,1,1)
        job_blue_collar=st.sidebar.slider('job_blue_collar',0,1,1)
        job_entrepreneur=st.sidebar.slider('job_entrepreneur',0,1,1)
        job_housemaid=st.sidebar.slider('job_housemaid',0,1,1)
        job_management=st.sidebar.slider('job_management',0,1,1)
        job_retired=st.sidebar.slider('job_retired',0,1,1)
        job_self_employed=st.sidebar.slider('job_self_employed',0,1,1)
        job_services=st.sidebar.slider('job_services',0,1,1)
        job_student=st.sidebar.slider('job_student',0,1,1)
        job_technician=st.sidebar.slider('job_technician',0,1,1)
        job_unemployed=st.sidebar.slider('job_unemployed',0,1,1)
        job_unknown=st.sidebar.slider('job_unknown',0,1,1)
        education_primary=st.sidebar.slider('education_primary',0,1,1)
        education_secondary=st.sidebar.slider('education_secondary',0,1,1)
        education_tertiary=st.sidebar.slider('education_tertiary',0,1,1)
        education_unknown=st.sidebar.slider('education_unknown',0,1,1)
        contact_cellular=st.sidebar.slider('contact_cellular',0,1,1)
        contact_telephone=st.sidebar.slider('contact_telephone',0,1,1)
        contact_unknown=st.sidebar.slider('contact_unknown',0,1,1)
        month_apr=st.sidebar.slider('month_apr',0,1,1)
        month_aug=st.sidebar.slider('month_aug',0,1,1)
        month_dec=st.sidebar.slider('month_dec',0,1,1)
        month_feb=st.sidebar.slider('month_feb',0,1,1)
        month_jan=st.sidebar.slider('month_jan',0,1,1)
        month_jul=st.sidebar.slider('month_jul',0,1,1)
        month_jun=st.sidebar.slider('month_jun',0,1,1)
        month_mar=st.sidebar.slider('month_mar',0,1,1)
        month_may=st.sidebar.slider('month_may',0,1,1)
        month_nov=st.sidebar.slider('month_nov',0,1,1)
        month_oct=st.sidebar.slider('month_oct',0,1,1)
        month_sep=st.sidebar.slider('month_sep',0,1,1)
        poutcome_failure=st.sidebar.slider('poutcome_failure',0,1,1)
        poutcome_other=st.sidebar.slider('poutcome_other',0,1,1)
        poutcome_success=st.sidebar.slider('poutcome_success',0,1,1)
        poutcome_unknown=st.sidebar.slider('poutcome_unknown',0,1,1)        
    data = {
            'age':age,
            'marital':marital,
            'default':default,
            'balance':balance,
            'housing':housing,
            'loan':loan,
            'day':day,
            'duration':duration,
            'campaign':campaign,
            'pdays':pdays,
            'previous':previous,
            'job_admin':job_admin,
            'job_blue_collar':job_blue_collar,
            'job_entrepreneur':job_entrepreneur,
            'job_housemaid':job_housemaid,
            'job_management':job_management,
            'job_retired':job_retired,
            'job_self_employed':job_self_employed,
            'job_services':job_services,
            'job_student':job_student,
            'job_technician':job_technician,
            'job_unemployed':job_unemployed,
            'job_unknown':job_unknown,
            'education_primary':education_primary,
            'education_secondary':education_secondary,
            'education_tertiary':education_tertiary,
            'education_unknown':education_unknown,
            'contact_cellular':contact_cellular,
            'contact_telephone':contact_telephone,
            'contact_unknown':contact_unknown,
            'month_apr':month_apr,
            'month_aug':month_aug,
            'month_dec':month_dec,
            'month_feb':month_feb,
            'month_jan':month_jan,
            'month_jul':month_jul,
            'month_jun':month_jun,
            'month_mar':month_mar,
            'month_may':month_may,
            'month_nov':month_nov,
            'month_oct':month_oct,
            'month_sep':month_sep,
            'poutcome_failure':poutcome_failure,
            'poutcome_other':poutcome_other,
            'poutcome_success':poutcome_success,
            'poutcome_unknown':poutcome_unknown
            }
    features = pd.DataFrame(data, index=[0])
    return features


feature_imp = st.sidebar.selectbox('Mau pake features important?',('Ya','Tidak'))
# st.write('You selected:', feature_imp)

if feature_imp == 'Tidak':
    st.warning('Kamu lagi ga pake fitur important ya..!, kalo kamu mau pake, aktifin di sidebar ya :), tapi kalo kamu pake, Machine Learning akan lebih berat sehingga loadingnya bakal lebih lama')
else:
    st.info('Kamu lagi pake Features Important, ingat Machine learning jadi lebih berat, loading bakal lebih lama ya!.')
st.write('---')

df = user_input_features() 

# Main Panel

# load the model from disk
from joblib import dump, load
# import xgboost as xgb


# result = loaded_model.score(X_test, Y_test)
model = load(open('ml_model/rfcmodel_deposito.pkl', 'rb'))
# model = RandomForestRegressor()
# model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

#Applying anti-log to transform into the normal values
# prediction = np.exp(prediction)-1

st.header('Perkiraan Deposito')

if prediction.item(0) == 1:
    st.info('###### Pelanggan dengan karakteristik yang telah dipilih diperkirakan akan melakukan deposito')
else:
    st.error('###### Pelanggan dengan karakteristik yang telah dipilih diperkirakan tidak akan melakukan deposito')
st.write('---')

# Print specified input parameters
st.header('Parameter input dipilih')
st.write(df)
st.write('---')



# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap

if feature_imp == 'Ya':
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    st.set_option('deprecation.showPyplotGlobalUse', False) #untuk memastikan fitur berfungsi baik di versi terbaru
    st.header('Feature Importance')
    st.write('---')
    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
# else:
#     st.warning('---')




