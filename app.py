# app.py
# 환승연애4 숏폼 vs 롱폼 참여 분석 Streamlit 앱

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="환승연애4 | 숏폼 vs 롱폼 참여 분석",
    layout="wide"
)

st.title("환승연애4 유튜브 콘텐츠 분석")
st.caption(
    "동일한 방송 클립임에도 불구하고, "
    "숏폼과 롱폼 포맷에 따라 사용자 참여 양상이 어떻게 달라지는지 탐색한다."
)

# ---------------------------------
# 데이터 로드
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("2237001_강선우_최종 데이터셋.xlsx")

df = load_data()

# 컬럼 정리
df["video_type"] = df["video_type"].str.lower().str.strip()
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
df["comments_engagement"] = pd.to_numeric(df["comments_engagement"], errors="coerce")
df["likes_engagement"] = pd.to_numeric(df["likes_engagement"], errors="coerce")

# ---------------------------------
# 사이드바 필터
# ---------------------------------
st.sidebar.header("필터 설정")

video_filter = st.sidebar.selectbox(
    "영상 타입 선택",
    ["전체", "shorts", "long"]
)

filtered_df = df.copy()
if video_filter != "전체":
    filtered_df = filtered_df[filtered_df["video_type"] == video_filter]

filtered_df = filtered_df.dropna(
    subset=["duration", "comments_engagement", "likes_engagement"]
)

# ---------------------------------
# 핵심 지표 (KPI)
# ---------------------------------
st.subheader("핵심 참여 지표")

col1, col2, col3 = st.columns(3)

col1.metric(
    "평균 댓글 참여율",
    round(filtered_df["comments_engagement"].mean(), 6)
)

col2.metric(
    "평균 좋아요 참여율",
    round(filtered_df["likes_engagement"].mean(), 6)
)

col3.metric(
    "영상 개수",
    len(filtered_df)
)

# ---------------------------------
# 시각화 1: 댓글 참여율 분포
# ---------------------------------
st.subheader("댓글 참여율 분포")

fig1, ax1 = plt.subplots()

if video_filter == "전체":
    sns.histplot(
        data=filtered_df,
        x="comments_engagement",
        hue="video_type",
        bins=30,
        ax=ax1
    )
else:
    sns.histplot(
        data=filtered_df,
        x="comments_engagement",
        bins=30,
        ax=ax1
    )

ax1.set_xlabel("comments_engagement")
ax1.set_ylabel("count")
ax1.set_title("댓글 참여율 분포")

st.pyplot(fig1)

st.caption(
    "해석: 동일한 방송 클립임에도 불구하고, "
    "숏폼과 롱폼 간 댓글 참여율의 분포 차이가 관찰된다. "
    "특히 롱폼 콘텐츠에서 상대적으로 높은 참여율을 보이는 영상이 일부 존재한다."
)

# ---------------------------------
# 시각화 2: 영상 길이 vs 댓글 참여율
# ---------------------------------
st.subheader("영상 길이와 댓글 참여율의 관계")

fig2, ax2 = plt.subplots()

if video_filter == "전체":
    sns.scatterplot(
        data=filtered_df,
        x="duration",
        y="comments_engagement",
        hue="video_type",
        ax=ax2
    )
else:
    sns.scatterplot(
        data=filtered_df,
        x="duration",
        y="comments_engagement",
        ax=ax2
    )

ax2.set_xlabel("영상 길이 (초)")
ax2.set_ylabel("댓글 참여율")
ax2.set_title("영상 길이와 댓글 참여율의 관계")

st.pyplot(fig2)

st.caption(
    "해석: 롱폼은 사용자의 선택에 의해 시청되는 포맷이라는 특성상, "
    "영상 길이가 길어질수록 댓글 참여율이 상대적으로 높아지는 경향이 관찰된다. "
    "이는 숏폼과 롱폼의 소비 맥락 차이를 시사한다."
)

# ---------------------------------
# 데이터 미리보기
# ---------------------------------
with st.expander("분석에 사용된 데이터 확인"):
    st.dataframe(filtered_df.reset_index(drop=True))
