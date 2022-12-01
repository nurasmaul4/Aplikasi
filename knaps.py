import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("DATA MINING")
st.write("""# Prediksi Penurunan/Peningkatan Jumlah Kasus Gizi Buruk""")
st.write("By: Nur Asmaul Khusna - 200411100012")
upload_data, Deskripsi_dataset, preporcessing, modeling, implementation = st.tabs(["Upload Data", "Description", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
        
with Deskripsi_dataset:
    st.write("""# Deskripsi""")
    st.write("Malnutrisi terus menjadi alasan yang membuat anak-anak jauh lebih rentan terhadap penyakit dan kematian.")
    st.write("Ada 4 jenis malnutrisi: wasting, stunting, underweight, dan overweight.")
    st.write("Wasting Berat - % anak usia 0–59 bulan yang berada di bawah minus tiga standar deviasi dari rata-rata berat badan terhadap tinggi badan.")
    st.write("1. Wasting – Sedang dan parah: % anak usia 0–59 bulan yang berada di bawah minus dua standar deviasi dari median berat badan- untuk-tinggi.")
    st.write("2. Kegemukan – Sedang dan berat: % usia 0-59 bulan yang berada di atas dua standar deviasi dari rata-rata berat badan terhadap tinggi badan.")
    st.write("3. Stunting – Sedang dan berat: % anak usia 0–59 bulan yang berada di bawah minus dua standar deviasi dari median tinggi-untuk-usia.")
    st.write("4. Kekurangan berat badan – Sedang dan berat: % anak usia 0–59 bulan yang berada di bawah minus dua standar deviasi dari rata-rata berat badan-untuk-usia.")

with preporcessing:
    st.write("""# Preprocessing""")
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
   
    df =pd.read_csv("https://raw.githubusercontent.com/nurasmaul4/Aplikasi/main/country-wise-average%20(2).csv")
    average=pd.read_csv("https://raw.githubusercontent.com/nurasmaul4/Aplikasi/main/country-wise-average%20(2).csv")
    average.head()
    
    st.write("average.describe")
    average.describe()
    
    x=average.iloc[:,1:]
    x
    
    from sklearn.impute import KNNImputer
    knn=KNNImputer()
    X=knn.fit_transform(x)
    X=pd.DataFrame(X)
    
    X.columns=["Income Classification","Severe Wasting","Wasting", "Overweight", "Stunting","Underweight","U5 Population ('000s)"]
    X.describe()
    
    country=average.iloc[:,0:1]
    X["country"]=country
    X
    
    st.write("Grouping is an option")
    X.groupby(by="Income Classification").mean()[["Overweight"]]
    
    st.write("Lower income level have lower overweight percentages")
    X.groupby(by="Income Classification").mean()[["Underweight"]]
    
    st.write("jika tingkat pendapatan meningkat, tingkat Underweight menurun drastis. Negara-negara miskin benar-benar memiliki masalah pasokan makanan")
    X[X["Income Classification"]<=0]["country"]
    
    st.write("Perang dan Teror dan Masalah Air dan tingkat pendapatan memiliki korelasi yang kuat.")
    X[X["Income Classification"]==3]["country"]
    
    
    with modeling:
        st.write("Modelling")
        
    with implementation:
        st.write("implementasi")

