# app.py
# 환승연애4 숏폼 vs 롱폼 참여 분석 Streamlit 앱 (엑셀 컬럼명 맞춤 버전)
# 실행:
#   pip install streamlit pandas matplotlib seaborn openpyxl
#   streamlit run app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="환승연애4 | 숏폼 vs 롱폼 참여 분석",
    layout="wide"
)

st.title("환승연애4 유튜브 콘텐츠 분석: 숏폼 vs 롱폼 참여 양상")
st.caption("동일 클립 기반 숏폼/롱폼 콘텐츠의 참여(댓글/좋아요) 양상 차이를 탐색합니다.")

# ---------------------------------
# Constants (너가 지정한 '최종' 컬럼명만 사용)
# ---------------------------------
COL_TYPE = "type"                 # video_type 대신
COL_DURATION = "duration_final"   # duration 대신
COL_DATE = "date_final"           # date 대신
COL_COMMENTS = "comments_engagement"
COL_LIKES = "likes_engagement"

REQUIRED_COLS = [COL_TYPE, COL_DURATION, COL_DATE, COL_COMMENTS, COL_LIKES]

# ---------------------------------
# Load data
# ---------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

DATA_PATH = "2237001_강선우_최종 데이터셋.xlsx"

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

# 컬럼명 공백 제거(숨은 공백 때문에 KeyError 나는 경우 방지)
df.columns = df.columns.astype(str).str.strip()

# 필수 컬럼 존재 확인
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"❌ 필수 컬럼이 누락되었습니다: {missing}")
    st.write("현재 컬럼 목록:", list(df.columns))
    st.stop()

# ---------------------------------
# Cleaning / type normalization
# ---------------------------------
# type 컬럼 정리: shorts/long 형태로 통일
df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip().str.lower()
df[COL_TYPE] = df[COL_TYPE].replace({
    "short": "shorts",
    "short-form": "shorts",
    "shortform": "shorts",
    "reels": "shorts",
    "long-form": "long",
    "longform": "long",
    "video": "long",
})

# 숫자형 변환
df[COL_DURATION] = pd.to_numeric(df[COL_DURATION], errors="coerce")
df[COL_COMMENTS] = pd.to_numeric(df[COL_COMMENTS], errors="coerce")
df[COL_LIKES] = pd.to_numeric(df[COL_LIKES], errors="coerce")

# 날짜 변환 (date_final만 사용)
df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

# 분석에 필요한 핵심 컬럼 결측 제거
base_df = df.dropna(subset=[COL_TYPE, COL_DURATION, COL_COMMENTS, COL_LIKES])

if base_df.empty:
    st.warning("분석에 필요한 핵심 컬럼에 결측치가 너무 많아 시각화할 데이터가 없습니다.")
    st.stop()

# ---------------------------------
# Sidebar filters
# ---------------------------------
st.sidebar.header("필터")

type_choice = st.sidebar.selectbox("영상 타입 선택", ["전체", "shorts", "long"], index=0)

# date_final 기반 날짜 필터(선택 기능)
use_date_filter = st.sidebar.checkbox("date_final로 기간 필터 사용", value=False)

filtered_df = base_df.copy()

if type_choice != "전체":
    filtered_df = filtered_df[filtered_df[COL_TYPE] == type_choice]

if use_date_filter:
    # date_final 결측 제외 후 범위 계산
    date_df = filtered_df.dropna(subset=[COL_DATE]).copy()
    if date_df.empty:
        st.sidebar.warning("선택된 조건에서 date_final 값이 없어 기간 필터를 적용할 수 없습니다.")
    else:
        min_date = date_df[COL_DATE].min().date()
        max_date = date_df[COL_DATE].max().date()
        start_date, end_date = st.sidebar.date_input(
            "기간 선택 (date_final 기준)",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        # 안전하게 정렬
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        filtered_df = filtered_df.dropna(subset=[COL_DATE])
        filtered_df = filtered_df[
            (filtered_df[COL_DATE].dt.date >= start_date) &
            (filtered_df[COL_DATE].dt.date <= end_date)
        ]

if filtered_df.empty:
    st.warning("선택한 필터 조건에서 데이터가 없습니다. 필터를 조정해 주세요.")
    st.stop()

# ---------------------------------
# KPI cards
# ---------------------------------
st.subheader("핵심 지표 (필터 반영)")

avg_comments = filtered_df[COL_COMMENTS].mean()
avg_likes = filtered_df[COL_LIKES].mean()
n_videos = len(filtered_df)

c1, c2, c3 = st.columns(3)
c1.metric("평균 댓글 참여율", f"{avg_comments:.6f}")
c2.metric("평균 좋아요 참여율", f"{avg_likes:.6f}")
c3.metric("영상 개수", f"{n_videos:,}")

# ---------------------------------
# Visualizations
# ---------------------------------
left, right = st.columns(2)

# 1) Histogram: comments engagement distribution
with left:
    st.subheader("댓글 참여율 분포 (히스토그램)")

    fig, ax = plt.subplots()
    if type_choice == "전체":
        sns.histplot(
            data=filtered_df,
            x=COL_COMMENTS,
            hue=COL_TYPE,
            bins=30,
            kde=False,
            ax=ax
        )
        ax.legend(title="type")
    else:
        sns.histplot(
            data=filtered_df,
            x=COL_COMMENTS,
            bins=30,
            kde=False,
            ax=ax
        )

    ax.set_xlabel(COL_COMMENTS)
    ax.set_ylabel("count")
    ax.set_title("댓글 참여율 분포")
    st.pyplot(fig, clear_figure=True)

    st.caption(
        "해석: 동일 클립 기반이더라도 포맷(type)에 따라 댓글 참여율의 분포가 달라질 수 있다. "
        "특정 구간(낮은 참여율)에 몰림이 크다면, 일부 콘텐츠만 상대적으로 높은 반응을 얻었을 가능성이 있다."
    )

# 2) Scatter: duration_final vs comments_engagement
with right:
    st.subheader("영상 길이(duration_final) vs 댓글 참여율 (산점도)")

    fig, ax = plt.subplots()
    if type_choice == "전체":
        sns.scatterplot(
            data=filtered_df,
            x=COL_DURATION,
            y=COL_COMMENTS,
            hue=COL_TYPE,
            ax=ax
        )
        ax.legend(title="type")
    else:
        sns.scatterplot(
            data=filtered_df,
            x=COL_DURATION,
            y=COL_COMMENTS,
            ax=ax
        )

    ax.set_xlabel(COL_DURATION)
    ax.set_ylabel(COL_COMMENTS)
    ax.set_title("duration_final과 댓글 참여율 관계")
    st.pyplot(fig, clear_figure=True)

    st.caption(
        "해석: 숏폼은 노출 기반 소비가 강하고, 롱폼은 선택 시청 성격이 강해 참여(댓글/좋아요) 양상이 다르게 나타날 수 있다. "
        "또한 duration_final 구간별로 참여율이 달라지는지 함께 관찰할 수 있다."
    )

# ---------------------------------
# Data preview
# ---------------------------------
with st.expander("필터 적용 데이터 보기"):
    show_cols = [COL_DATE, COL_TYPE, COL_DURATION, COL_COMMENTS, COL_LIKES]
    st.dataframe(filtered_df[show_cols].reset_index(drop=True))
