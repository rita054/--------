# -*- coding: utf-8 -*-
"""
TF-IDF + Cosine 推薦（僅支援：情境選單）
- 使用 SVM Emotion Model（取自 emotion_svm_trainer.py 的訓練方式）
- 讀取 Excel：artist, song, text, track_id, popularity, human_label
- human_label 允許像 "Q1(開心/興奮)" 這種格式，會自動萃取成 Q1~Q4

Usage:
    1) 把 new_songs_for_human_labeling.xlsx 放同層（或自行改 excel_path）
    2) python TFIDF_cosine_download_choose_svm.py
"""

import os
import re
import numpy as np
import pandas as pd

# NLP / TF-IDF
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Model: SVM (probability=True 才能 predict_proba)
from sklearn.svm import SVC

# Similarity
from sklearn.metrics.pairwise import cosine_similarity


# ===== 0) 全域設定：情緒維度（固定順序）=====
EMOTIONS = ["Q1", "Q2", "Q3", "Q4"]  # Q1開心/興奮, Q2憤怒/緊張, Q3悲傷/痛苦, Q4放鬆/平靜


# ===== 1) 讀取資料 =====
def load_songs_from_excel(excel_path: str) -> pd.DataFrame:
    """
    讀取歌曲 training data（Excel），並確保 text 欄位為字串且非空。
    預期欄位：artist, song, text, track_id, popularity, human_label
    """
    df = pd.read_excel(excel_path)
    df["text"] = df["text"].fillna("").astype(str)

    # human_label 可能是 NaN 或各種字串格式：Q1 / Q1(開心/興奮) / q2 ...
    if "human_label" not in df.columns:
        df["human_label"] = np.nan

    df["label_clean"] = df["human_label"].apply(extract_q_label)
    return df


def extract_q_label(val):
    """把 human_label 萃取成 'Q1'~'Q4'，找不到就回 NaN。"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.search(r"(Q[1-4])", s, re.IGNORECASE)
    return m.group(1).upper() if m else np.nan


# ===== 2) 歌詞清理 =====
def get_stop_words() -> set:
    """
    NLTK 英文停用詞 + 歌詞常見口水詞/段落標記
    """
    sw = set(stopwords.words("english")).union({
        "oh", "yeah", "hey", "la", "da", "ooh", "ah", "na", "ha",
        "chorus", "verse", "intro", "outro", "bridge", "refrain",
        "im", "youre", "hes", "shes", "theyre", "aint", "gonna",
        "wanna", "gotta", "feat", "ft", "yo", "uh", "mmm"
    })
    return sw


def clean_lyrics(text: str, stop_words: set) -> str:
    """
    清理流程：
    1) 小寫
    2) 去掉 [Chorus] 這種括號段落
    3) 去標點、去數字
    4) tokenize
    5) 移除停用詞、長度<=1 的 token
    """
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)     # 去掉 [chorus] [verse] ...
    text = re.sub(r"[^\w\s]", " ", text)     # 去標點
    text = re.sub(r"\d+", " ", text)         # 去數字

    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)


# ===== 3) 建 TF-IDF =====
def build_tfidf_matrices_for_train_and_rec(
    train_df: pd.DataFrame,
    rec_df: pd.DataFrame,
):
    """
    同時為「訓練集」與「推薦候選集」建立 TF-IDF 特徵（共用同一個 vectorizer）。

    - train_df：用於訓練（含 human_label）
    - rec_df：用於推薦（不一定有 human_label）

    回傳：
    - vectorizer
    - X_train: train_df 的 TF-IDF
    - X_rec: rec_df 的 TF-IDF
    - train_df_clean, rec_df_clean（含 clean_text）
    - stop_words（供 debug/一致清理）
    """
    stop_words = get_stop_words()

    train_df = train_df.copy()
    rec_df = rec_df.copy()

    train_df["clean_text"] = train_df["text"].apply(lambda t: clean_lyrics(t, stop_words))
    rec_df["clean_text"] = rec_df["text"].apply(lambda t: clean_lyrics(t, stop_words))

    # ⚠️ 很重要：vectorizer 以「train+rec」一起 fit，避免推薦集大量 OOV（詞不在 vocab）
    fit_corpus = pd.concat([train_df["clean_text"], rec_df["clean_text"]], axis=0).fillna("")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    vectorizer.fit(fit_corpus)

    X_train = vectorizer.transform(train_df["clean_text"])
    X_rec = vectorizer.transform(rec_df["clean_text"])

    return vectorizer, X_train, X_rec, train_df, rec_df, stop_words


# ===== 4) 訓練情緒模型：SVM =====
def train_emotion_model_svm(X, y: pd.Series) -> SVC:
    """
    用 linear SVM 做 multi-class 分類，並可輸出 predict_proba 作為情緒分數向量。
    y 預期值：Q1/Q2/Q3/Q4
    """
    clf = SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(X, y)
    print("clf.classes_ =", clf.classes_)
    return clf


def predict_emotion_scores(clf: SVC, X) -> np.ndarray:
    """
    回傳每首歌的情緒機率向量 shape = (n_songs, 4)
    順序對齊 EMOTIONS（Q1,Q2,Q3,Q4）
    """
    proba = clf.predict_proba(X)  # shape: (n, n_classes_in_training)

    # 重新排列成固定 EMOTIONS 順序（缺類別就補 0）
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    scores = np.zeros((proba.shape[0], len(EMOTIONS)), dtype=float)

    for j, emo in enumerate(EMOTIONS):
        if emo in class_to_idx:
            scores[:, j] = proba[:, class_to_idx[emo]]
        else:
            scores[:, j] = 0.0

    return scores


# ===== 5) 情境 -> 情緒規則 =====
SCENARIO_TO_EMOTION = {
    "深夜放鬆": [0.05, 0.05, 0.40, 0.50],
    "運動健身": [0.70, 0.30, 0.00, 0.00],
    "旅行": [0.55, 0.10, 0.05, 0.30],
    "失戀難過": [0.02, 0.25, 0.70, 0.03],
    "專注讀書": [0.05, 0.10, 0.05, 0.80],
    "派對狂歡": [0.90, 0.07, 0.01, 0.01],
    "通勤": [0.30, 0.00, 0.00, 0.70],
}


def get_scenario_vector(scenario: str) -> np.ndarray:
    if scenario not in SCENARIO_TO_EMOTION:
        raise ValueError(f"未知情境：{scenario}。請先在 SCENARIO_TO_EMOTION 裡新增規則。")

    q = np.array(SCENARIO_TO_EMOTION[scenario], dtype=float).reshape(1, -1)
    s = q.sum()
    if s <= 0:
        raise ValueError("情境向量加總不可為 0。")
    return q / s



# ===== 6) 推薦：給定 query 情緒向量 q（1,4）-> cosine -> Top-N =====
def recommend_top_n_by_query_vector(
    df: pd.DataFrame,
    emotion_scores: np.ndarray,
    q: np.ndarray,
    top_n: int = 20,
    max_per_artist: int = 3
) -> pd.DataFrame:
    sims = cosine_similarity(q, emotion_scores).flatten()  # (n,)

    out = df.copy()
    out["sim"] = sims

    # tie-break 用 popularity（若沒有就補 0）
    if "popularity" not in out.columns:
        out["popularity"] = 0

    out = out.sort_values(["sim", "popularity"], ascending=[False, False])

    picked = []
    artist_count = {}

    for _, row in out.iterrows():
        a = row.get("artist", "")
        if artist_count.get(a, 0) >= max_per_artist:
            continue
        picked.append(row)
        artist_count[a] = artist_count.get(a, 0) + 1
        if len(picked) >= top_n:
            break

    cols = ["artist", "song", "track_id", "popularity", "sim"]
    cols = [c for c in cols if c in out.columns]
    return pd.DataFrame(picked)[cols].reset_index(drop=True)


def recommend_top_n(
    df: pd.DataFrame,
    emotion_scores: np.ndarray,
    scenario: str,
    top_n: int = 20,
    max_per_artist: int = 3
) -> pd.DataFrame:
    q = get_scenario_vector(scenario)
    return recommend_top_n_by_query_vector(df, emotion_scores, q, top_n, max_per_artist)


# ===== 7) 主程式 =====
def main():
    # === 1) 訓練用資料（含 human_label） ===
    train_excel_path = "new_songs_for_human_labeling.xlsx"

    # === 2) 推薦候選歌曲資料（不一定有 human_label） ===
    rec_excel_path = "lyrics_with_spotify_meta_merged.xlsx"

    if not os.path.exists(train_excel_path):
        print(f"找不到訓練檔案：{train_excel_path}")
        print("請把檔案放到同一層，或修改 main() 裡的 train_excel_path。")
        return

    if not os.path.exists(rec_excel_path):
        print(f"找不到推薦候選檔案：{rec_excel_path}")
        print("請把檔案放到同一層，或修改 main() 裡的 rec_excel_path。")
        return

    # 讀入兩份資料
    train_df = load_songs_from_excel(train_excel_path)
    rec_df = load_songs_from_excel(rec_excel_path)

    # 共用同一個 TF-IDF 向量器（避免推薦集 OOV）
    vectorizer, X_train, X_rec, train_df_clean, rec_df_clean, stop_words = build_tfidf_matrices_for_train_and_rec(
        train_df=train_df,
        rec_df=rec_df
    )

    # === 3) 訓練 SVM（僅使用 train_df 裡有標註的資料）===
    has_label = train_df_clean["label_clean"].notna().any()
    clf = None

    if not has_label:
        print("目前訓練檔沒有有效 human_label（Q1~Q4）可訓練 SVM 模型。")
        print("將使用均勻情緒向量做推薦（效果會較差）。")
        emotion_scores_rec = np.ones((len(rec_df_clean), len(EMOTIONS)), dtype=float) / len(EMOTIONS)
    else:
        labeled_mask = train_df_clean["label_clean"].notna().to_numpy()
        X_labeled = X_train[labeled_mask]
        y = train_df_clean.loc[labeled_mask, "label_clean"].astype(str)

        clf = train_emotion_model_svm(X_labeled, y)

        # === 4) 對「推薦候選集」產生情緒分數向量（Q1~Q4）===
        emotion_scores_rec = predict_emotion_scores(clf, X_rec)

    # ====== 只支援：情境選單 ======
    scenarios = list(SCENARIO_TO_EMOTION.keys())
    print("請選擇情境：")
    for i, s in enumerate(scenarios, start=1):
        print(f"{i}. {s}")

    while True:
        choice = input("輸入選項編號：").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(scenarios):
                scenario = scenarios[idx - 1]
                break
        print("輸入無效，請輸入清單中的編號。")

    # 用推薦候選集 rec_df_clean + emotion_scores_rec 做推薦
    q = get_scenario_vector(scenario)
    rec = recommend_top_n_by_query_vector(
        df=rec_df_clean,
        emotion_scores=emotion_scores_rec,
        q=q,
        top_n=15,
        max_per_artist=2
    )

    print(f"=== 情境：{scenario} 的 Top-N 推薦（候選集：{rec_excel_path}）===")
    print(rec)


if __name__ == "__main__":
    main()
