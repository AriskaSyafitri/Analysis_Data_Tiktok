import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from scipy.sparse import hstack
from datetime import datetime
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="📊 Model Popularitas TikTok", layout="wide")

# Memuat data (hanya sekali)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/tiktok_scrapper.csv")
        return df
    except Exception as e:
        st.error(f"Kesalahan saat memuat data: {e}")
        return pd.DataFrame()

# Pra-pemrosesan Data
def preprocess_data(df):
    df.dropna(inplace=True)
    le_name = LabelEncoder()
    df['authorMeta.name_encoded'] = le_name.fit_transform(df['authorMeta.name'])
    
    le_music = LabelEncoder()
    df['musicMeta.musicName_encoded'] = le_music.fit_transform(df['musicMeta.musicName'])

    df['text_length'] = df['text'].apply(len)
    df['hashtags_str'] = df['text'].apply(lambda x: ' '.join(re.findall(r"#\w+", str(x))))
    df['createTimeISO'] = pd.to_datetime(df['createTimeISO'])
    df['hour'] = df['createTimeISO'].dt.hour
    df['minute'] = df['createTimeISO'].dt.minute
    df['second'] = df['createTimeISO'].dt.second
    df['day'] = df['createTimeISO'].dt.dayofweek 

    df['total_interactions'] = (df['diggCount'] + df['shareCount'] + 
                                df['commentCount'] + df['playCount'])
    df['is_popular'] = (df['total_interactions'] >= 5000000).astype(int)

    tfidf = TfidfVectorizer(max_features=100) 
    hashtag_tfidf = tfidf.fit_transform(df['text'])

    features = hstack((
        hashtag_tfidf,
        np.array(df[['authorMeta.name_encoded', 'musicMeta.musicName_encoded', 
                      'videoMeta.duration', 'hour', 'minute', 'second', 'text_length']])
    ))
    return df, features, df['is_popular'], tfidf, le_name, le_music

# Evaluasi model tanpa melatih ulang
def evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Menampilkan metrik
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{accuracy:.2f}")
    col2.metric("Presisi", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.round(2)
    st.subheader("Laporan Klasifikasi")
    st.dataframe(report_df)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Matriks Kebingungan")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor='#000000')
    sns.heatmap(cm, annot=True, fmt='d', cmap='inferno',
                xticklabels=['Tidak Populer', 'Populer'],
                yticklabels=['Tidak Populer', 'Populer'], ax=ax_cm)
    ax_cm.set_facecolor('#000000')
    ax_cm.set_title('Matriks Kebingungan', color='white')
    ax_cm.tick_params(colors='white')
    for spine in ax_cm.spines.values():
        spine.set_color('white')
    st.pyplot(fig_cm)

# Prediksi popularitas konten
def predict_content(model, tfidf, text, author, music, duration, waktu):
    text = str(text)
    text_length = len(text)
    hour = waktu.hour
    minute = waktu.minute
    second = waktu.second
    
    author_encoded = st.session_state.le_name.transform([str(author)])[0] if str(author) in st.session_state.le_name.classes_ else -1
    music_encoded = st.session_state.le_music.transform([str(music)])[0] if str(music) in st.session_state.le_music.classes_ else -1
    
    tfidf_matrix = tfidf.transform([text])
    features = hstack((
        tfidf_matrix,
        np.array([[author_encoded, music_encoded, duration, hour, minute, second, text_length]])
    ))
    
    prediction = model.predict(features)
    return prediction[0]

# Prediksi konten dalam jumlah banyak
def predict_bulk(model, tfidf, df_input):
    df_input['text'] = df_input['text'].fillna('').astype(str) 
    df_input['text_length'] = df_input['text'].apply(len)

    df_input['createTimeISO'] = pd.to_datetime(df_input['createTimeISO'], errors='coerce')
    df_input['hour'] = df_input['createTimeISO'].dt.hour
    df_input['minute'] = df_input['createTimeISO'].dt.minute
    df_input['second'] = df_input['createTimeISO'].dt.second

    df_input['authorMeta.name_encoded'] = df_input['authorMeta.name'].apply(
        lambda x: st.session_state.le_name.transform([str(x)])[0]
        if str(x) in st.session_state.le_name.classes_ else -1
    )
    df_input['musicMeta.musicName_encoded'] = df_input['musicMeta.musicName'].apply(
        lambda x: st.session_state.le_music.transform([str(x)])[0]
        if str(x) in st.session_state.le_music.classes_ else -1
    )

    tfidf_matrix = tfidf.transform(df_input['text'])

    features = hstack((tfidf_matrix, np.array(df_input[[
        'authorMeta.name_encoded', 'musicMeta.musicName_encoded',
        'videoMeta.duration', 'hour', 'minute', 'second', 'text_length'
    ]])))

    df_input['status_popularitas'] = model.predict(features)
    df_input['status_popularitas'] = df_input['status_popularitas'].map({
        1: "🔥 Populer", 0: "❄️ Tidak Populer"
    })
    return df_input

# Aplikasi utama
def main():
    st.sidebar.title("📊 Dashboard Sistem")
    if st.sidebar.button("📈 EDA dan Visualisasi Data"): 
        st.session_state.section = 'EDA'
    if st.sidebar.button("🧠 Model Evaluasi Konten"): 
        st.session_state.section = 'Model'
    if st.sidebar.button("📁 Informasi Data TikTok"): 
        st.session_state.section = 'Data'
    if st.sidebar.button("🎯 Popularitas Konten TikTok"): 
        st.session_state.section = 'Prediksi'
    
    st.sidebar.markdown("---")
    st.sidebar.write("🎬 Dashboard Popularitas TikTok")

    if 'section' not in st.session_state:
        st.session_state.section = 'EDA'

    df = load_data()  # Memuat data hanya sekali
    if df.empty:
        return  # Hentikan jika pemuatan data gagal

    df, X, y, tfidf, le_name, le_music = preprocess_data(df)
    
    if st.session_state.section == 'EDA':
        st.header("1. Analisis Data Eksploratif")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Distribusi Interaksi", "🔗 Korelasi", "📈 Hubungan Antar Fitur", "⏰ Analisis Berdasarkan Waktu",  "🎵 Analisis Kategori Musik"])

        with tab1:
            st.subheader("Distribusi Interaksi")
            col1, col2 = st.columns(2)
            colors = ['#39FF14', '#FF073A', '#05FFA1', '#FFD300']
            metrics = ['diggCount', 'shareCount', 'playCount', 'commentCount']
            titles = ['Like', 'Share', 'Play', 'Komentar']
            for metric, title, color, col in zip(metrics, titles, colors, [col1, col1, col2, col2]):
                fig = px.histogram(df, x=metric, nbins=20, title=title, color_discrete_sequence=[color])
                fig.update_layout(
                    plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white', title_font_size=18)
                col.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Korelasi Interaksi")
            fig_corr, ax = plt.subplots(figsize=(4, 3), facecolor='#000000')
            sns.heatmap(df[metrics].corr(), annot=True, cmap='magma', annot_kws={"size": 8}, ax=ax)
            ax.set_facecolor('#000000')
            ax.tick_params(colors='white', labelsize=8)
            ax.set_title('Korelasi antar Interaksi', color='white', fontsize=10)
            st.pyplot(fig_corr)

        with tab3:
            st.subheader("Visualisasi Hubungan Antar Fitur")
            scatter_cols = st.columns(3)
            pairs = [('shareCount', 'diggCount'), ('playCount', 'diggCount'), ('commentCount', 'shareCount')]
            scatter_titles = ['Share vs Like', 'Play vs Like', 'Komentar vs Share']
            for (x, y_), title, color, col in zip(pairs, scatter_titles, colors[:3], scatter_cols):
                fig_sc = px.scatter(df, x=x, y=y_, title=title, color_discrete_sequence=[color])
                fig_sc.update_layout(
                    plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white', title_font_size=18)
                col.plotly_chart(fig_sc, use_container_width=True)

        with tab4:
            st.subheader("⏰ Analisis Berdasarkan Waktu dan Kategori Musik")
            hour_df = df.groupby(['hour', 'is_popular']).size().unstack(fill_value=0)
            hour_fig = px.line(hour_df, x=hour_df.index, y=hour_df.columns,
                                title='Distribusi Popularitas Konten Berdasarkan Jam',
                                labels={'x': 'Jam', 'value': 'Jumlah Konten', 'variable': 'Popularitas'},
                                color_discrete_sequence=['#39FF14', '#FF073A'])
            hour_fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
            st.plotly_chart(hour_fig, use_container_width=True)

            day_df = df.groupby(['day', 'is_popular']).size().unstack(fill_value=0)
            day_fig = px.bar(day_df, x=day_df.index, y=day_df.columns,
                            title='Distribusi Popularitas Konten Berdasarkan Hari',
                            labels={'x': 'Hari', 'value': 'Jumlah Konten', 'variable': 'Popularitas'},
                            color_discrete_sequence=['#39FF14', '#FF073A'])
            day_fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
            st.plotly_chart(day_fig, use_container_width=True)

        with tab5:
            st.subheader("🎵 Analisis Berdasarkan Kategori Musik")
            music_interactions = df.groupby('musicMeta.musicName')['total_interactions'].sum().reset_index()
            top_music = music_interactions.sort_values(by='total_interactions', ascending=False).head(10)

            music_fig = px.bar(top_music, x='musicMeta.musicName', y='total_interactions',
                                title='10 Kategori Musik Paling Banyak Digunakan',
                                labels={'musicMeta.musicName': 'Kategori Musik', 'total_interactions': 'Total Interaksi'},
                                color='total_interactions', color_continuous_scale=px.colors.sequential.Viridis)
            music_fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
            st.plotly_chart(music_fig, use_container_width=True)

    elif st.session_state.section == 'Model':
        st.header("2. Evaluasi Model Random Forest")

        # Load model & encoder hasil training dari notebook
        model = joblib.load("model/rf_model.pkl")
        le_name = joblib.load("model/le_name.pkl")
        le_music = joblib.load("model/le_music.pkl")
        tfidf = joblib.load("model/tfidf_hashtag.pkl")

        st.session_state.model = model
        st.session_state.le_name = le_name
        st.session_state.le_music = le_music
        st.session_state.tfidf = tfidf

        # Buat ulang fitur dari data agar sesuai
        tfidf_matrix = tfidf.transform(df['text'])  # Gunakan 'text', bukan 'hashtags_str'
        features = hstack((
            tfidf_matrix,
            np.array(df[['authorMeta.name_encoded', 'musicMeta.musicName_encoded',
                         'videoMeta.duration', 'hour', 'minute', 'second', 'text_length']])
        ))
        X = features
        y = df['is_popular']

        evaluate_model(model, X, y)

    elif st.session_state.section == 'Data':
        st.header("3. Tinjau Dataset")
        st.dataframe(df)
        with st.expander("📌 Statistik Deskriptif"):
            st.dataframe(df.describe())

    elif st.session_state.section == 'Prediksi':
        st.header("4. Klasifikasi Popularitas Konten TikTok")
        if 'model' not in st.session_state:
            st.warning("⚠️ Model belum dilatih. Silakan jalankan terlebih dahulu bagian '🧠 Model Evaluasi Konten'.")
            return

        tab1, tab2, tab3 = st.tabs(["🔮 Hasil Uji Satu Konten", "📅 Hasil Uji Banyak Konten", "✍️ Input Manual"])

        with tab1:
            text = st.text_area("Deskripsi Konten")
            author = st.text_input("Nama Kreator")
            music = st.text_input("Musik yang Digunakan")
            duration = st.slider("Durasi video (detik)", 0, 300)
            waktu_jam = st.time_input("Waktu Unggah", datetime.strptime("12:01:00", "%H:%M:%S").time())
            if st.button("🚀 Uji Popularitas Konten"):
                result = predict_content(st.session_state.model, st.session_state.tfidf, text, author, music, duration, waktu_jam)
                if result == 1:
                    st.success("❤️‍🔥 Konten ini  **Populer**!")
                else:
                    st.warning("⚠️ Konten ini  **Tidak Populer**.")

        with tab2:
            uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
            if uploaded_file:
                df_input = pd.read_csv(uploaded_file)
                predicted_df = predict_bulk(st.session_state.model, st.session_state.tfidf, df_input)
                st.dataframe(predicted_df[['text', 'authorMeta.name', 'musicMeta.musicName', 'videoMeta.duration', 'createTimeISO', 'status_popularitas']])

        with tab3:
            rows = st.number_input("Jumlah Baris Input Manual", min_value=1, max_value=10, value=1)
            manual_data = []
            for i in range(rows):
                st.markdown(f"### Konten {i+1}")
                text = st.text_area(f"Deskripsi Konten {i+1}", key=f"text_{i}")
                author = st.text_input(f"Nama Author {i+1}", key=f"author_{i}")
                music = st.text_input(f"Nama Musik {i+1}", key=f"music_{i}")
                duration = st.number_input(f"Durasi Video (detik) {i+1}", min_value=1, key=f"durasi_{i}")
                waktu = st.time_input(f"Waktu Unggah {i+1}", key=f"time_{i}")
                waktu_iso = datetime.now().replace(hour=waktu.hour, minute=waktu.minute, second=waktu.second)
                manual_data.append({
                    'text': text,
                    'authorMeta.name': author,
                    'musicMeta.musicName': music,
                    'videoMeta.duration': duration,
                    'createTimeISO': waktu_iso
                })
            if st.button("🚀 Uji Popularitas Data Manual"):
                df_manual = pd.DataFrame(manual_data)
                predicted_df = predict_bulk(st.session_state.model, st.session_state.tfidf, df_manual)
                st.dataframe(predicted_df[['text', 'authorMeta.name', 'musicMeta.musicName', 'videoMeta.duration', 'createTimeISO', 'status_popularitas']])

if __name__ == '__main__':
    main()
