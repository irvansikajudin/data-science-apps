import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.write("""
# App Prediksi Harga Mobil
Aplikasi ini memprediksi **Harga Mobil**!, 
[Link dokumentasi Machine Learning di Github](https://github.com/irvansikajudin/Car-Price-Prediction)
""")


# Loads the Boston House Price Dataset
# boston = datasets.load_boston()
X = pd.read_csv('dataset/car_price_after_preprocessing_tanpa_target.csv')
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.info('### Pilih Parameter Input')

def user_input_features():
    Levy=st.sidebar.slider('Levy',0.0,7528.0,7528.0)
    prodyear=st.sidebar.slider('prodyear',1943,2020,2020)
    Leather_interior=st.sidebar.slider('Leather_interior',0,1,1)
    enginevolume=st.sidebar.slider('enginevolume',0.0,10.8,10.8)
    # Mileage=st.sidebar.slider('Mileage',0,2147483647,2147483647)
    Mileage=st.sidebar.slider('Mileage X 100',0,21474836,1800000)
    # Karena Mileage nilainya sangat besar, maka nilai dikonversikan ke 1% persen utk dapat ditampilkan,
    # setelah itu baru di konversikan lagi ke nilai semula dengan di kali 100
    Mileage = Mileage * 100
    Wheel=st.sidebar.slider('Wheel',0,1,1)
    Airbags=st.sidebar.slider('Airbags',0,16,16)
    Turbo=st.sidebar.slider('Turbo',0,1,1)
    Category_Cabriolet=st.sidebar.slider('Category_Cabriolet',0,1,1)
    Category_Coupe=st.sidebar.slider('Category_Coupe',0,1,1)
    Category_Goods_wagon=st.sidebar.slider('Category_Goods_wagon',0,1,1)
    Category_Hatchback=st.sidebar.slider('Category_Hatchback',0,1,1)
    Category_Jeep=st.sidebar.slider('Category_Jeep',0,1,1)
    Category_Limousine=st.sidebar.slider('Category_Limousine',0,1,1)
    Category_Microbus=st.sidebar.slider('Category_Microbus',0,1,1)
    Category_Minivan=st.sidebar.slider('Category_Minivan',0,1,1)
    Category_Pickup=st.sidebar.slider('Category_Pickup',0,1,1)
    Category_Sedan=st.sidebar.slider('Category_Sedan',0,1,1)
    Category_Universal=st.sidebar.slider('Category_Universal',0,1,1)
    Fuel_type_CNG=st.sidebar.slider('Fuel_type_CNG',0,1,1)
    Fuel_type_Diesel=st.sidebar.slider('Fuel_type_Diesel',0,1,1)
    Fuel_type_Hybrid=st.sidebar.slider('Fuel_type_Hybrid',0,1,1)
    Fuel_type_Hydrogen=st.sidebar.slider('Fuel_type_Hydrogen',0,1,1)
    Fuel_type_LPG=st.sidebar.slider('Fuel_type_LPG',0,1,1)
    Fuel_type_Petrol=st.sidebar.slider('Fuel_type_Petrol',0,1,1)
    Fuel_type_Plug_in_Hybrid=st.sidebar.slider('Fuel_type_Plug_in_Hybrid',0,1,1)
    Gear_box_type_Automatic=st.sidebar.slider('Gear_box_type_Automatic',0,1,1)
    Gear_box_type_Manual=st.sidebar.slider('Gear_box_type_Manual',0,1,1)
    Gear_box_type_Tiptronic=st.sidebar.slider('Gear_box_type_Tiptronic',0,1,1)
    Gear_box_type_Variator=st.sidebar.slider('Gear_box_type_Variator',0,1,1)
    Drive_wheels_4x4=st.sidebar.slider('Drive_wheels_4x4',0,1,1)
    Drive_wheels_Front=st.sidebar.slider('Drive_wheels_Front',0,1,1)
    Drive_wheels_Rear=st.sidebar.slider('Drive_wheels_Rear',0,1,1)
    Doors_2_3=st.sidebar.slider('Doors_2_3',0,1,1)
    Doors_4_5=st.sidebar.slider('Doors_4_5',0,1,1)
    Doors_morethan_5=st.sidebar.slider('Doors_morethan_5',0,1,1)
    Color_Beige=st.sidebar.slider('Color_Beige',0,1,1)
    Color_Black=st.sidebar.slider('Color_Black',0,1,1)
    Color_Blue=st.sidebar.slider('Color_Blue',0,1,1)
    Color_Brown=st.sidebar.slider('Color_Brown',0,1,1)
    Color_Carnelian_red=st.sidebar.slider('Color_Carnelian_red',0,1,1)
    Color_Golden=st.sidebar.slider('Color_Golden',0,1,1)
    Color_Green=st.sidebar.slider('Color_Green',0,1,1)
    Color_Grey=st.sidebar.slider('Color_Grey',0,1,1)
    Color_Orange=st.sidebar.slider('Color_Orange',0,1,1)
    Color_Pink=st.sidebar.slider('Color_Pink',0,1,1)
    Color_Purple=st.sidebar.slider('Color_Purple',0,1,1)
    Color_Red=st.sidebar.slider('Color_Red',0,1,1)
    Color_Silver=st.sidebar.slider('Color_Silver',0,1,1)
    Color_Skyblue=st.sidebar.slider('Color_Skyblue',0,1,1)
    Color_White=st.sidebar.slider('Color_White',0,1,1)
    Color_Yellow=st.sidebar.slider('Color_Yellow',0,1,1)
    data = {
            'Levy':Levy,
            'prodyear':prodyear,
            'Leather_interior':Leather_interior,
            'enginevolume':enginevolume,
            'Mileage':Mileage,
            'Wheel':Wheel,
            'Airbags':Airbags,
            'Turbo':Turbo,
            'Category_Cabriolet':Category_Cabriolet,
            'Category_Coupe':Category_Coupe,
            'Category_Goods_wagon':Category_Goods_wagon,
            'Category_Hatchback':Category_Hatchback,
            'Category_Jeep':Category_Jeep,
            'Category_Limousine':Category_Limousine,
            'Category_Microbus':Category_Microbus,
            'Category_Minivan':Category_Minivan,
            'Category_Pickup':Category_Pickup,
            'Category_Sedan':Category_Sedan,
            'Category_Universal':Category_Universal,
            'Fuel_type_CNG':Fuel_type_CNG,
            'Fuel_type_Diesel':Fuel_type_Diesel,
            'Fuel_type_Hybrid':Fuel_type_Hybrid,
            'Fuel_type_Hydrogen':Fuel_type_Hydrogen,
            'Fuel_type_LPG':Fuel_type_LPG,
            'Fuel_type_Petrol':Fuel_type_Petrol,
            'Fuel_type_Plug_in_Hybrid':Fuel_type_Plug_in_Hybrid,
            'Gear_box_type_Automatic':Gear_box_type_Automatic,
            'Gear_box_type_Manual':Gear_box_type_Manual,
            'Gear_box_type_Tiptronic':Gear_box_type_Tiptronic,
            'Gear_box_type_Variator':Gear_box_type_Variator,
            'Drive_wheels_4x4':Drive_wheels_4x4,
            'Drive_wheels_Front':Drive_wheels_Front,
            'Drive_wheels_Rear':Drive_wheels_Rear,
            'Doors_2_3':Doors_2_3,
            'Doors_4_5':Doors_4_5,
            'Doors_morethan_5':Doors_morethan_5,
            'Color_Beige':Color_Beige,
            'Color_Black':Color_Black,
            'Color_Blue':Color_Blue,
            'Color_Brown':Color_Brown,
            'Color_Carnelian_red':Color_Carnelian_red,
            'Color_Golden':Color_Golden,
            'Color_Green':Color_Green,
            'Color_Grey':Color_Grey,
            'Color_Orange':Color_Orange,
            'Color_Pink':Color_Pink,
            'Color_Purple':Color_Purple,
            'Color_Red':Color_Red,
            'Color_Silver':Color_Silver,
            'Color_Skyblue':Color_Skyblue,
            'Color_White':Color_White,
            'Color_Yellow':Color_Yellow
            }
    features = pd.DataFrame(data, index=[0])
    return features


feature_imp = st.sidebar.selectbox('Mau pake features important?',('Tidak', 'Ya'))
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
model = load(open('ml_model/dtreemodel_car_prediction.pkl', 'rb'))

# result = loaded_model.score(X_test, Y_test)

# model = RandomForestRegressor()
# model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

#Applying anti-log to transform into the normal values
prediction = np.exp(prediction)-1

st.header('Perkiraan harga mobil')
# st.write(prediction)

a = 'Mobil dengan karakteristik yang telah dipilih diperkirakan memiliki harga : '
b = round(prediction.item(0),2)
c = a, b
c = str(c)
c = c.replace("('", "")
c = c.replace("'", "")
c = c.replace(")", "")
c = c.replace(",", "")
st.info(c)
st.write('---')

# Print specified input parameters
st.header('Parameter input dipilih')
st.write(df)
st.write('---')


# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap

if feature_imp == 'Ya':
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    st.set_option('deprecation.showPyplotGlobalUse', False) #untuk memastikan fitur berfungsi baik di versi terbaru
    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
# else:
#     st.warning('---')



