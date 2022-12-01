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
    import numpy as np
    import pandas as pd
    
    # Display settings
    pd.options.display.max_rows = 400
    pd.options.display.max_columns = 100
    pd.options.display.float_format = "{:.2f}".format

    random_state = 42
    np.random.seed(random_state)
    
    # Suppress warnings
    import warnings; warnings.filterwarnings('ignore')

    data = pd.read_csv('https://raw.githubusercontent.com/nurasmaul4/Aplikasi/main/country-wise-average%20(2).csv')
    data_by_country = pd.read_csv('https://raw.githubusercontent.com/nurasmaul4/Aplikasi/main/country-wise-average%20(2).csv')
    data.drop(['Unnamed: 0','ISO code','Survey Year','Source','Report Author','Notes','Short Source'], axis=1, inplace=True)

    def income_map(val):
        mapper = {0:'Low Income', 1:'Lower Middle Income', 2:'Upper Middle Income',3:'High Income'}
        return mapper[val]
    def lldc_map(val):
        mapper = {0:'Others', 2:'SIDS', 1:'LLDC'}
        return mapper[val]

    data['Income Classification'] = data['Income Classification'].apply(income_map)
    data['LLDC or SID2'] = data['LLDC or SID2'].apply(lldc_map)
    
    data.head()
    
    data.columns
    
    data.info()
    
    data.describe().T
    
    st.write("Check missing values in the dataframe")
    data.isnull().sum()
    
    columns = list(['Severe Wasting', 'Wasting','Overweight', 'Stunting', 'Underweight'])

    print('Descriptive Stats before imputation for columns with missing values: \n', '--'*35)
    display(data[columns].describe().T)

    data['Wasting'].fillna(data['Wasting'].mean(), inplace=True)
    data['Severe Wasting'].fillna(data['Severe Wasting'].mean(), inplace=True)
    data['Overweight'].fillna(data['Overweight'].mean(), inplace=True)
    data['Stunting'].fillna(data['Stunting'].mean(), inplace=True)
    data['Underweight'].fillna(data['Underweight'].mean(), inplace=True)

    print('Descriptive Stats after imputation: \n', '--'*35)
    display(data[columns].describe().T)
    
    st.write("Univariate Analysis")
    # Functions that will help us with EDA plot
    def odp_plots(df, col):
        f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))
    
    # Boxplot to check outliers
    sns.boxplot(x = col, data = df, ax = ax1, orient = 'v', color = 'darkslategrey')
    
    # Distribution plot with outliers
    sns.distplot(df[col], ax = ax2, color = 'teal', fit = norm).set_title(f'{col} with outliers')
    
    # Removing outliers, but in a new dataframe
    upperbound, lowerbound = np.percentile(df[col], [1, 99])
    y = pd.DataFrame(np.clip(df[col], upperbound, lowerbound))
    
    # Distribution plot without outliers
    sns.distplot(y[col], ax = ax3, color = 'tab:orange', fit = norm).set_title(f'{col} without outliers')
    
    kwargs = {'fontsize':14, 'color':'black'}
    ax1.set_title(col + ' Boxplot Analysis', **kwargs)
    ax1.set_xlabel('Box', **kwargs)
    ax1.set_ylabel(col + ' Values', **kwargs)

    
    
    st.write("Distribution plots")
    # Outlier, distribution for columns with outliers
    boxplotcolumns = ['Severe Wasting', 'Wasting', 'Overweight', 'Stunting',
                      'Underweight']
    for cols in boxplotcolumns:
        Q3 = data[cols].quantile(0.75)
        Q1 = data[cols].quantile(0.25)
        IQR = Q3 - Q1

        print(f'{cols.capitalize()} column', '--'*40)
        count = len(data.loc[(data[cols] < (Q1 - 1.5 * IQR)) | (data[cols] > (Q3 + 1.5 * IQR))])
        print(f'no of records with outliers values: {count}')

        display(data.loc[(data[cols] < (Q1 - 1.5 * IQR)) | (data[cols] > (Q3 + 1.5 * IQR))].head())
        print(f'EDA for {cols.capitalize()} column', '--'*40)
        odp_plots(data, cols)

    del cols, IQR, boxplotcolumns
    
    st.write("Multivariate Analysis")
    corr = data.corr()
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask,  linewidths = .5 )#, annot = True)
    
    # Filter for correlation value greater than threshold
    sort = corr.abs().unstack()
    sort = sort.sort_values(kind = "quicksort", ascending = False)
    display(sort[(sort > 0.7) & (sort < 1)])
    
    st.write("Negara mana yang menunjukkan persentase Underweight tertinggi? ---> Bangladesh")
    country = data.loc[:,['Country','Underweight']]
    country['percunder'] = country.groupby('Country')['Underweight'].transform('max')
    country = country.drop('Underweight',axis=1).drop_duplicates().sort_values('percunder', ascending=False).head()

    fig = px.pie(country, names='Country', values='percunder', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    fig.show()
    
    st.write("Negara mana yang menunjukkan persentase Kegemukan tertinggi? ---> Albania")
    country = data.loc[:,['Country','Overweight']]
    country['percunder'] = country.groupby('Country')['Overweight'].transform('max')
    country = country.drop('Overweight',axis=1).drop_duplicates().sort_values('percunder', ascending=False).head()

    fig = px.pie(country, names='Country', values='percunder', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    fig.show()
    
    st.write("Kelas pendapatan mana yang memiliki persentase underweight tertinggi? ---> Pendapatan Menengah Bawah")
    f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))
    sns.distplot(data['Underweight'], ax=ax1)

    df_LM = data.loc[data['Income Classification'] == 'Lower Middle Income']
    df_UM = data.loc[data['Income Classification'] == 'Upper Middle Income']
    df_Low = data.loc[data['Income Classification'] == 'Low Income']
    df_High = data.loc[data['Income Classification'] == 'High Income']

    sns.distplot( df_LM['Underweight'],ax = ax2 , color = 'r')
    sns.distplot( df_UM['Underweight'],ax = ax2, color = 'g')
    sns.distplot( df_Low['Underweight'],ax = ax2, color = 'b')
    sns.distplot( df_High['Underweight'],ax = ax2, color = 'y')
    
    df = data.loc[:,['Income Classification','Underweight']]
    df['maxunder'] = df.groupby('Income Classification')['Underweight'].transform('mean')
    df = df.drop('Underweight', axis=1).drop_duplicates()
    df = data.loc[:,['Income Classification','Underweight']]
    df['maxunder'] = df.groupby('Income Classification')['Underweight'].transform('mean')
    df = df.drop('Underweight', axis=1).drop_duplicates()

    fig = sns.barplot(data=df, x='Income Classification', y='maxunder')
    fig.set(xticklabels = ['LM', 'UM', 'Low', "High"])
    plt.show()
    
    df = data.loc[:,['Income Classification','Underweight']]
    df['maxunder'] = df.groupby('Income Classification')['Underweight'].transform('max')
    df = df.drop('Underweight', axis=1).drop_duplicates()

    fig = px.pie(df, names='Income Classification', values='maxunder', template='seaborn')
    fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label", showlegend=False)
    fig.show()
    
    st.write("Persentase underweight di Negara Terbelakang vs Negara Maju")
    f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))
    df_with_LDC = data.loc[data['LDC'] == 1]
    df_with_DC = data.loc[data['LDC'] == 0]

    sns.distplot(data['Underweight'], ax=ax1)
    sns.distplot( df_with_LDC['Underweight'],ax = ax2 , color = 'r')
    sns.distplot( df_with_DC['Underweight'],ax = ax2, color = 'g')

    df = data.loc[:,['LIFD','Underweight']]
    df['maxunder'] = df.groupby('LIFD')['Underweight'].transform('mean')
    df = df.drop('Underweight', axis=1).drop_duplicates()

    fig = sns.barplot(data=df, x='LIFD', y='maxunder', ax=ax3)
    fig.set(xticklabels = ['Not LIFD', 'LIFD'])
    plt.show()
    
    st.write("Persentase Negara Kekurangan Pangan Berpenghasilan Rendah")
    f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))
    df_with_LIFD = data.loc[data['LIFD'] == 1]
    df_with_NLIFD = data.loc[data['LIFD'] == 0]

    sns.distplot(data['Underweight'], ax=ax1)
    sns.distplot( df_with_LIFD['Underweight'],ax = ax2 , color = 'r')
    sns.distplot( df_with_NLIFD['Underweight'],ax = ax2, color = 'g')

    df = data.loc[:,['LIFD','Underweight']]
    df['maxunder'] = df.groupby('LIFD')['Underweight'].transform('mean')
    df = df.drop('Underweight', axis=1).drop_duplicates()
    df = data.loc[:,['LIFD','Underweight']]
    df['maxunder'] = df.groupby('LIFD')['Underweight'].transform('mean')
    df = df.drop('Underweight', axis=1).drop_duplicates()

    fig = sns.barplot(data=df, x='LIFD', y='maxunder')
    fig.set(xticklabels = ['Not LIFD', 'LIFD'])
    plt.show()
    
    st.write("Analisis Underweight menurut Kelompok Pendapatan")
    data["Income Classification"].value_counts()
    
    st.write("Negara Berkembang yang Terkurung Daratan vs Negara Berkembang Pulau Kecil vs Lainnya ---> Lainnya")
    df = data.loc[:,['LLDC or SID2','Underweight']]
    df['maxunder'] = df.groupby('LLDC or SID2')['Underweight'].transform('max')
    df = df.drop('Underweight', axis=1).drop_duplicates()

    fig = px.pie(df, names='LLDC or SID2', values='maxunder', template='seaborn')
    fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label", showlegend=False)
    fig.show()
    
    st.write("Severe Wasting")
    st.write("Ini adalah % anak usia 0–59 bulan yang berada di bawah minus tiga standar deviasi dari rata-rata berat badan terhadap tinggi badan")
    sns.kdeplot(data=data['Severe Wasting'], shade=True)
    plt.title('Distribution of Sever Wasting percentages in countries')
    plt.show()
    
    st.write("Kita dapat melihat dari plot distribusi bahwa nilai persentase o setinggi 12% di beberapa negara. Dari scatter plot dapat diketahui bahwa persentase wasting parah yang tinggi sebagian besar ditemukan untuk ukuran sampel survei yang rendah.")
    
    st.write("Severe Wasting - Wasting - Overweight - Underweight")
    sns.pairplot(data[['Severe Wasting','Overweight','Underweight', 'Stunting']])
    plt.show()
    
    st.write("penduduk negara tersebut yang berusia di bawah 5 tahun")
    sns.kdeplot(data=data['U5 Population (\'000s)'], shade=True)
    plt.title('Distribution of U5 Population')
    plt.show()
    

    with modeling:
        st.write("Modelling")
        
    with implementation:
        st.write("implementasi")

