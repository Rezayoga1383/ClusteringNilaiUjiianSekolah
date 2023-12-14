import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import skfuzzy as fuzz

def fuzzy():
    # Dataset
    # st.header("Dataset")
    # st.write("Data awal sebelum melakukan preprocessing")
    data = pd.read_csv('https://gist.githubusercontent.com/Rezayoga1383/db61676b040382c6c15438c6524e7375/raw/6eccc1e5c8b79c0f6d233ae51d49efc0ec543213/nilai2.csv')
    data = data.drop(['nama'], axis=1)
    # st.write(data)

    # Normalisasi
    # st.header("Normalisasi")
    # normalisasi dataset
    numeric_data = data[['matematika', 'biologi', 'fisika','kimia','bahasa inggris','pai','pkn','bahasa indonesia','sejarah','seni budaya','penjasorkes','prakarya','bahasa madura']]
    minmax = MinMaxScaler()
    normalized_numeric = minmax.fit_transform(numeric_data)
    # Konversi hasil normalisasi ke dalam DataFrame Pandas
    normalized_df = pd.DataFrame(normalized_numeric, columns=numeric_data.columns)
    # st.write(normalized_df)

    # data akan direduksi menjadi dua dimensi.
    # st.header("Mereduksi Data")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_df)
    # Membuat DataFrame untuk hasil PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    # st.write(pca_df)
    df_fuzzy = pca_df.copy()
    # Ubah DataFrame menjadi numpy array
    data_array = df_fuzzy.values
    # Tentukan jumlah cluster
    n_clusters = 3
    # menjaga konsistensi
    np.random.seed(42)
    # Terapkan Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_array.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    # Ambil label cluster untuk setiap data point
    cluster_membership = np.argmax(u, axis=0)
    # Tambahkan label cluster ke DataFrame
    df_fuzzy['Cluster'] = cluster_membership
    st.write(df_fuzzy)
    # Menghitung Silhouette Coefficient
    silhouette_avg = silhouette_score(pca_df, df_fuzzy['Cluster'])
    st.write("Silhoutte Score = ",silhouette_avg)

def hirarki():
    # Dataset
    # st.header("Dataset")
    # st.write("Data awal sebelum melakukan preprocessing")
    data = pd.read_csv('https://gist.githubusercontent.com/Rezayoga1383/db61676b040382c6c15438c6524e7375/raw/6eccc1e5c8b79c0f6d233ae51d49efc0ec543213/nilai2.csv')
    data = data.drop(['nama'], axis=1)
    # st.write(data)

    # Normalisasi
    # st.header("Normalisasi")
    # normalisasi dataset
    numeric_data = data[['matematika', 'biologi', 'fisika','kimia','bahasa inggris','pai','pkn','bahasa indonesia','sejarah','seni budaya','penjasorkes','prakarya','bahasa madura']]
    minmax = MinMaxScaler()
    normalized_numeric = minmax.fit_transform(numeric_data)
    # Konversi hasil normalisasi ke dalam DataFrame Pandas
    normalized_df = pd.DataFrame(normalized_numeric, columns=numeric_data.columns)
    # st.write(normalized_df)

    # data akan direduksi menjadi dua dimensi.
    # st.header("Mereduksi Data")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_df)
    # Membuat DataFrame untuk hasil PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    # st.write(pca_df)
    df_hierar = pca_df.copy()
    # menjaga konsistensi
    np.random.seed(42)
    # Melakukan Hierarchical Clustering dengan 3 kluster
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical.fit(df_hierar)
    # Tambahkan label cluster ke DataFrame PCA
    df_hierar['Cluster'] = hierarchical.labels_
    st.write(df_hierar)
    silhouette_avg = silhouette_score(pca_df, df_hierar['Cluster'])
    st.write("Nilai Silhouette Score",silhouette_avg)

def means():
    # Dataset
    # st.header("Dataset")
    # st.write("Data awal sebelum melakukan preprocessing")
    data = pd.read_csv('https://gist.githubusercontent.com/Rezayoga1383/db61676b040382c6c15438c6524e7375/raw/6eccc1e5c8b79c0f6d233ae51d49efc0ec543213/nilai2.csv')
    data = data.drop(['nama'], axis=1)
    # st.write(data)

    # Normalisasi
    # st.header("Normalisasi")
    # normalisasi dataset
    numeric_data = data[['matematika', 'biologi', 'fisika','kimia','bahasa inggris','pai','pkn','bahasa indonesia','sejarah','seni budaya','penjasorkes','prakarya','bahasa madura']]
    minmax = MinMaxScaler()
    normalized_numeric = minmax.fit_transform(numeric_data)
    # Konversi hasil normalisasi ke dalam DataFrame Pandas
    normalized_df = pd.DataFrame(normalized_numeric, columns=numeric_data.columns)
    # st.write(normalized_df)

    # data akan direduksi menjadi dua dimensi.
    # st.header("Mereduksi Data")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_df)
    # Membuat DataFrame untuk hasil PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    # st.write(pca_df)
    df_kmeans = pca_df.copy()
    # mengambil nilai dari pca (tanpa nama kolom)
    x = df_kmeans.iloc[:,  0:2].values
    # Melakukan K-Means Clustering dengan 3 kluster
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    # Fit model pada data PCA
    kmeans.fit(x)
    # Tambahkan label cluster ke DataFrame PCA
    df_kmeans['Cluster'] = kmeans.labels_
    st.write(df_kmeans)

    silhouette_avg = silhouette_score(pca_df, df_kmeans['Cluster'])
    st.write("Silhouette Score =",silhouette_avg)

def proses():
    # Dataset
    st.header("Dataset")
    st.write("Data awal sebelum melakukan preprocessing")
    data = pd.read_csv('https://gist.githubusercontent.com/Rezayoga1383/db61676b040382c6c15438c6524e7375/raw/6eccc1e5c8b79c0f6d233ae51d49efc0ec543213/nilai2.csv')
    data = data.drop(['nama'], axis=1)
    st.write(data)

    # Normalisasi
    st.header("Normalisasi")
    # normalisasi dataset
    numeric_data = data[['matematika', 'biologi', 'fisika','kimia','bahasa inggris','pai','pkn','bahasa indonesia','sejarah','seni budaya','penjasorkes','prakarya','bahasa madura']]
    minmax = MinMaxScaler()
    normalized_numeric = minmax.fit_transform(numeric_data)
    # Konversi hasil normalisasi ke dalam DataFrame Pandas
    normalized_df = pd.DataFrame(normalized_numeric, columns=numeric_data.columns)
    st.write(normalized_df)

    # data akan direduksi menjadi dua dimensi.
    st.header("Mereduksi Data")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_df)
    # Membuat DataFrame untuk hasil PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    st.write(pca_df)

    # Menghitung SSE untuk menentukan jumlah kluster
    st.header("Grafik Elbow Method")
    SSE = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
        kmeans.fit(pca_df)
        SSE.append(kmeans.inertia_)
        print(i,SSE)
    # Plot Elbow Curve
    plt.plot(range(1, 11), SSE, marker='o', linestyle='-')
    plt.title('Elbow Curve')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('SSE')
    st.pyplot(plt)

def plot_clusters(data, num_clusters):
    df_kmeans = data.copy()
    x = df_kmeans.iloc[:, 0:2].values
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(x)
    df_kmeans['Cluster'] = kmeans.labels_

    fig, ax = plt.subplots()  # Membuat figure dan axis baru
    colors = ['red', 'blue', 'green', 'black', 'orange', 'purple', 'pink', 'cyan']  # Warna untuk plot
    centroids_color = 'yellow'  # Warna centroids
    
    for cluster_num in range(num_clusters):
        ax.scatter(x[df_kmeans['Cluster'] == cluster_num, 0], x[df_kmeans['Cluster'] == cluster_num, 1],
                   s=100, c=colors[cluster_num], label=f'Cluster {cluster_num + 1}')
    
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100,
               c=centroids_color, label='Centroids')
    ax.set_title('Cluster')
    ax.legend()
    
    # Menampilkan plot dengan st.pyplot()
    st.pyplot(fig)
    st.write(df_kmeans)



# def submit(num_clusters,data):
#     # Dataset
#     st.write("Hasil Pelabelan K-Means")
#     data = pd.read_csv('https://gist.githubusercontent.com/Rezayoga1383/db61676b040382c6c15438c6524e7375/raw/6eccc1e5c8b79c0f6d233ae51d49efc0ec543213/nilai2.csv')
#     data = data.drop(['nama'], axis=1)

#     # normalisasi dataset
#     numeric_data = data[['matematika', 'biologi', 'fisika','kimia','bahasa inggris','pai','pkn','bahasa indonesia','sejarah','seni budaya','penjasorkes','prakarya','bahasa madura']]
#     minmax = MinMaxScaler()
#     normalized_numeric = minmax.fit_transform(numeric_data)
#     # Konversi hasil normalisasi ke dalam DataFrame Pandas
#     normalized_df = pd.DataFrame(normalized_numeric, columns=numeric_data.columns)
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(normalized_df)
#     # Membuat DataFrame untuk hasil PCA
#     pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    
#     df_kmeans = pca_df.copy()
#     # mengambil nilai dari pca (tanpa nama kolom)
#     x = df_kmeans.iloc[:,  0:2].values
#     # Melakukan K-Means Clustering dengan 3 kluster
#     kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
#     # Fit model pada data PCA
#     kmeans.fit(x)
#     # Tambahkan label cluster ke DataFrame PCA
#     df_kmeans['Cluster'] = kmeans.labels_
#     st.write(df_kmeans)

def main():
    
    # Tampilan menu
    st.title('Aplikasi Clustering Nilai Ujian Sekolah')

    menu_options = ["Implementasi", "Preprocessing", "K-Means","Hierarchical","Fuzzy"]
    selected_menu = st.selectbox('Pilih Menu', menu_options)

    # Menampilkan konten sesuai dengan pilihan menu
    if selected_menu == "Implementasi":
        st.header('Implementasi')
        # st.write('Selamat datang di beranda.')
        # Dataset
        # st.write("Dataset")
        data = pd.read_csv('https://gist.githubusercontent.com/Rezayoga1383/db61676b040382c6c15438c6524e7375/raw/6eccc1e5c8b79c0f6d233ae51d49efc0ec543213/nilai2.csv')
        data = data.drop(['nama'], axis=1)
        # st.write(data)

        # Normalisasi
        # st.write("Normalisasi")
        # normalisasi dataset
        numeric_data = data[['matematika', 'biologi', 'fisika','kimia','bahasa inggris','pai','pkn','bahasa indonesia','sejarah','seni budaya','penjasorkes','prakarya','bahasa madura']]
        minmax = MinMaxScaler()
        normalized_numeric = minmax.fit_transform(numeric_data)
        # Konversi hasil normalisasi ke dalam DataFrame Pandas
        normalized_df = pd.DataFrame(normalized_numeric, columns=numeric_data.columns)
        # st.write(normalized_df)

        # data akan direduksi menjadi dua dimensi.
        # st.write("Mereduksi Data Menjadi 2 Dimensi")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(normalized_df)
        # Membuat DataFrame untuk hasil PCA
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        # st.write(pca_df)

        # Menampilkan gambar di tengah halaman
        # file_path = "virtual/img/ujian.jpg"
        # st.image(file_path, width=310)

        # Slider untuk memilih jumlah kluster
        num_clusters = st.slider("Pilih Jumlah Kluster", min_value=2, max_value=8, value=3)

        # Mengatur tampilan tombol ke samping
        st.markdown(
            """
            <style>
            .stButton > button {
                width: 100%;
                text-align: center;
            }
            .button-container {
                display: flex;
                justify-content: space-between;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


        if st.button("Submit"):
            plot_clusters(pca_df,num_clusters)
        # if st.button("Proses"):
        #     proses()


    elif selected_menu == "Preprocessing":
        st.title('Preprocessing')
        st.write('Tahapan Preprocessing Dijelaskan di Bawah ini.')
        proses()

    elif selected_menu == "K-Means":
        st.title('K-Means')
        st.write('Hasil Dari Model K-Means Clustering.')
        means()
        
    
    elif selected_menu == "Hierarchical":
        st.title('Hierarchical')
        st.write('Hasil Dari Model Hierarchical.')
        hirarki()
    
    elif selected_menu == "Fuzzy":
        st.title('Fuzzy')
        st.write('Hasil Dari Model Fuzzy.')
        fuzzy()
    


if __name__ == "__main__":
    main()


