import streamlit as st
import pandas as pd
import numpy as np
import os
import re

# 頁面設定
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# 自定義CSS美化
st.markdown("""
<style>
    /* 主標題樣式 */
    .main-title {
        font-size: 2.5em;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5em;
        padding-bottom: 10px;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* 副標題樣式 */
    .sub-title {
        font-size: 1.1em;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2em;
        font-weight: 300;
    }
    
    /* 情境選擇器樣式 - 移除左邊藍色邊框 */
    .scenario-container {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 2em;
    }
    
    /* 歌曲列表樣式 */
    .song-item {
        padding: 12px 16px;
        margin: 6px 0;
        background: white;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        transition: all 0.2s ease;
        font-size: 1em;
    }
    
    .song-item:hover {
        border-color: #3B82F6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
        transform: translateY(-1px);
    }
    
    .song-title {
        font-weight: 600;
        color: #111827;
    }
    
    .song-artist {
        color: #6B7280;
        font-size: 0.95em;
    }
    
    /* 按鈕樣式 */
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1em;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    /* 統計卡片樣式 */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stats-number {
        font-size: 2em;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .stats-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# 標題
st.markdown('<div class="main-title">Music Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Select a scenario to get personalized song recommendations</div>', unsafe_allow_html=True)

# 情境選擇
st.markdown('<div class="scenario-container">', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    # 情境選擇器
    scenarios = {
        "late_night_relax": "Late Night Relax",
        "workout": "Workout & Exercise", 
        "road_trip": "Road Trip",
        "study_focus": "Study Focus",
        "heartbreak": "Heartbreak",
        "party": "Party & Celebration",
        "commute": "Daily Commute"
    }
    
    selected_key = st.selectbox(
        "Select Listening Scenario",
        list(scenarios.keys()),
        format_func=lambda x: scenarios[x],
        index=0
    )

with col2:
    # 空出空間對齊
    st.write("")

st.markdown('</div>', unsafe_allow_html=True)

# 推薦按鈕
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    get_recommendations = st.button(
        "Get Song Recommendations",
        type="primary",
        use_container_width=True
    )

# 分隔線
st.markdown("---")

# ===== 情境推薦邏輯 =====

# 情境情緒向量
SCENARIO_EMOTION_VECTORS = {
    "late_night_relax": [0.05, 0.05, 0.40, 0.50],
    "workout": [0.70, 0.30, 0.00, 0.00],
    "road_trip": [0.55, 0.10, 0.05, 0.30],
    "study_focus": [0.05, 0.10, 0.05, 0.80],
    "heartbreak": [0.02, 0.25, 0.70, 0.03],
    "party": [0.90, 0.07, 0.01, 0.01],
    "commute": [0.30, 0.00, 0.00, 0.70]
}

def extract_song_emotion(text):
    """從歌詞文本中提取情緒特徵"""
    text = str(text).lower()
    
    happy_words = ['happy', 'joy', 'love', 'smile', 'fun', 'party', 'dance', 'celebrate']
    angry_words = ['angry', 'hate', 'fight', 'rage', 'mad', 'furious', 'storm']
    sad_words = ['sad', 'cry', 'tear', 'hurt', 'pain', 'alone', 'miss', 'goodbye']
    calm_words = ['calm', 'peace', 'quiet', 'rest', 'sleep', 'dream', 'night', 'soft']
    
    happy_score = sum(text.count(word) for word in happy_words)
    angry_score = sum(text.count(word) for word in angry_words)
    sad_score = sum(text.count(word) for word in sad_words)
    calm_score = sum(text.count(word) for word in calm_words)
    
    scores = [happy_score, angry_score, sad_score, calm_score]
    total = sum(scores) + 0.001
    
    return [score / total for score in scores]

def calculate_similarity(emotion1, emotion2):
    """計算兩個情緒向量的餘弦相似度"""
    dot_product = sum(a * b for a, b in zip(emotion1, emotion2))
    norm1 = np.sqrt(sum(a * a for a in emotion1))
    norm2 = np.sqrt(sum(b * b for b in emotion2))
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def get_scenario_recommendations(df, scenario_key, top_n=15):
    """根據情境獲取推薦歌曲"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    scenario_vector = SCENARIO_EMOTION_VECTORS.get(scenario_key, [0.25, 0.25, 0.25, 0.25])
    
    similarities = []
    
    for idx, row in df.iterrows():
        text = str(row.get('text', '')) if 'text' in row else ''
        song_emotion = extract_song_emotion(text)
        similarity = calculate_similarity(song_emotion, scenario_vector)
        similarities.append(similarity)
    
    df_temp = df.copy()
    df_temp['similarity'] = similarities
    
    # 根據相似度排序
    df_temp = df_temp.sort_values('similarity', ascending=False)
    
    # 限制每位藝人的歌曲數量
    picked_indices = []
    artist_count = {}
    
    for idx, row in df_temp.iterrows():
        artist = str(row.get('artist', 'Unknown'))
        if artist_count.get(artist, 0) >= 3:
            continue
        
        picked_indices.append(idx)
        artist_count[artist] = artist_count.get(artist, 0) + 1
        
        if len(picked_indices) >= top_n:
            break
    
    return df_temp.loc[picked_indices]

# ===== 顯示推薦結果 =====
if get_recommendations:
    selected_scenario = scenarios[selected_key]
    
    # 檢查資料檔案
    excel_files = [
        "lyrics_with_spotify_meta_merged.xlsx",
        "new_songs_for_human_labeling.xlsx"
    ]
    
    data_file = None
    for file in excel_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file:
        try:
            # 讀取Excel檔案
            df = pd.read_excel(data_file)
            
            # 找出歌曲和歌手欄位
            song_col = None
            artist_col = None
            text_col = None
            
            # 檢查常見的欄位名稱
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['song', 'title', 'track_name', 'song_name', 'name']:
                    song_col = col
                elif col_lower in ['artist', 'singer', 'artist_name', 'performer']:
                    artist_col = col
                elif col_lower in ['text', 'lyrics', 'lyric', '歌詞']:
                    text_col = col
            
            # 如果找不到標準欄位，使用前幾欄
            if not song_col and len(df.columns) > 0:
                song_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            if not artist_col and len(df.columns) > 1:
                artist_col = df.columns[0]
            
            if not text_col:
                for col in df.columns:
                    if df[col].dtype == 'object' and len(df[col].astype(str).str.split()) > 5:
                        text_col = col
                        break
            
            if not text_col:
                df['text'] = ''
                text_col = 'text'
            
            # 進行情境推薦
            recommendations = get_scenario_recommendations(df, selected_key, top_n=20)
            
            # 顯示情境標題
            st.markdown(f"### Recommended Songs for **{selected_scenario}**")
            st.caption(f"Showing {len(recommendations)} personalized recommendations")
            
            # 統計卡片
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="stats-card">
                    <div class="stats-number">{}</div>
                    <div class="stats-label">Total Songs</div>
                </div>
                """.format(len(recommendations)), unsafe_allow_html=True)
            
            with col2:
                if artist_col:
                    unique_artists = recommendations[artist_col].nunique()
                    st.markdown("""
                    <div class="stats-card">
                        <div class="stats-number">{}</div>
                        <div class="stats-label">Unique Artists</div>
                    </div>
                    """.format(unique_artists), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="stats-card">
                        <div class="stats-number">-</div>
                        <div class="stats-label">Artists</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                # 使用情境名稱的第一個字
                scenario_word = selected_scenario.split()[0]
                st.markdown("""
                <div class="stats-card">
                    <div class="stats-number">{}</div>
                    <div class="stats-label">Scenario</div>
                </div>
                """.format(scenario_word), unsafe_allow_html=True)
            
            # 顯示推薦歌曲
            st.markdown("---")
            st.markdown("### Song List")
            
            # 建立可滾動的歌曲列表容器
            songs_container = st.container()
            
            with songs_container:
                for idx, row in recommendations.iterrows():
                    # 取得歌曲和歌手
                    song = str(row[song_col]) if pd.notnull(row[song_col]) else "Unknown Song"
                    artist = str(row[artist_col]) if artist_col and pd.notnull(row[artist_col]) else "Unknown Artist"
                    
                    # 顯示美觀的歌曲項目（移除分數顯示）
                    st.markdown(f"""
                    <div class="song-item">
                        <span class="song-title">{song}</span>
                        <span style="color: #9CA3AF; margin: 0 8px">|</span>
                        <span class="song-artist">{artist}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 移除底部統計資訊
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            # 顯示樣本歌曲
            st.markdown("### Sample Recommendations")
            show_sample_songs()
    
    else:
        st.warning("Data file not found. Showing sample songs.")
        show_sample_songs()

def show_sample_songs():
    """顯示範例歌曲"""
    st.caption("Displaying sample songs for demonstration")
    
    # 樣本資料
    sample_data = [
        ("Nocturne", "Jay Chou"),
        ("Fish", "Cheer Chen"),
        ("Stubborn", "Mayday"),
        ("A Little Happiness", "Hebe Tien"),
        ("Light Years Away", "G.E.M."),
        ("Love Confession Balloon", "Jay Chou"),
        ("Decent", "Yu Wenwen"),
        ("Superman", "G.E.M."),
        ("Stranger in the North", "Namewee"),
        ("Jump", "Mayday"),
        ("Rainbow", "Jay Chou"),
        ("Simple Love", "Jay Chou"),
        ("Sunshine After Rain", "Jolin Tsai"),
        ("Small Love Song", "Deserts Chang"),
        ("You Are Not Truly Happy", "Mayday")
    ]
    
    # 顯示樣本歌曲
    for song, artist in sample_data:
        st.markdown(f"""
        <div class="song-item">
            <span class="song-title">{song}</span>
            <span style="color: #9CA3AF; margin: 0 8px">|</span>
            <span class="song-artist">{artist}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.caption(f"Showing {len(sample_data)} sample songs")

# 側邊欄資訊
st.sidebar.markdown("## About")
st.sidebar.markdown("""
This system recommends songs based on emotional analysis of lyrics.

**Features:**
- Emotion-based matching
- Personalized recommendations

**How to use:**
1. Select a listening scenario
2. Click "Get Song Recommendations"
3. View your personalized playlist
""")

# 底部資訊
st.sidebar.markdown("---")