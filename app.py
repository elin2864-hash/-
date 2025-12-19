# app.py
# 환승연애4 숏폼 vs 롱폼 참여 분석 대시보드
# - 한글 깨짐 방지: Plotly 사용
# - 회차 기준 쌍비교: 같은 날짜 or ±1일 업로드 → 같은 회차로 간주
# - Top 10% 영상: 버튼 클릭 시 st.video로 앱 내 재생

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="환승연애4 | 숏폼 vs 롱폼 참여 분석", layout="wide")
st.title("환승연애4 유튜브 숏폼 vs 롱폼 참여 분석 대시보드")
st.caption("동일 클립 기반 콘텐츠의 포맷(type)에 따른 참여(댓글/좋아요) 양상 차이를 탐색합니다.")

# -----------------------------
# 컬럼명 (너 데이터셋 기준 고정)
# -----------------------------
COL_TYPE = "type"
COL_DURATION = "duration_final"
COL_DATE = "date_final"
COL_COMMENTS = "Comments_Engagement"
COL_LIKES = "Likes_Engagement"
COL_VIEWS = "viewCount"
COL_URL = "url"
COL_TITLE = "title"

REQUIRED_COLS = [COL_TYPE, COL_DURATION, COL_DATE, COL_COMMENTS, COL_LIKES, COL_URL]
DATA_PATH = "2237001_강선우_최종 데이터셋.xlsx"

# Plotly 폰트(브라우저 폰트 우선)
PLOTLY_FONT = dict(family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif", size=14)

# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

df = load_data(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"❌ 필수 컬럼이 누락되었습니다: {missing}")
    st.write("현재 컬럼 목록:", list(df.columns))
    st.stop()

# -----------------------------
# 전처리
# -----------------------------
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

df[COL_DURATION] = pd.to_numeric(df[COL_DURATION], errors="coerce")
df[COL_COMMENTS] = pd.to_numeric(df[COL_COMMENTS], errors="coerce")
df[COL_LIKES] = pd.to_numeric(df[COL_LIKES], errors="coerce")
if COL_VIEWS in df.columns:
    df[COL_VIEWS] = pd.to_numeric(df[COL_VIEWS], errors="coerce")

df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

base_df = df.dropna(subset=[COL_TYPE, COL_DURATION, COL_COMMENTS, COL_LIKES, COL_URL]).copy()
if base_df.empty:
    st.warning("분석에 필요한 핵심 값 결측이 많아 시각화할 데이터가 없습니다.")
    st.stop()

# 범례용 라벨
type_label_map = {"long": "롱폼(long)", "shorts": "숏폼(shorts)"}
base_df["type_label"] = base_df[COL_TYPE].map(type_label_map).fillna(base_df[COL_TYPE])

# 날짜(일 단위)
base_df["date_only"] = base_df[COL_DATE].dt.date

# -----------------------------
# Sidebar 필터
# -----------------------------
st.sidebar.header("필터")

type_choice = st.sidebar.selectbox("영상 타입 선택", ["전체", "shorts", "long"], index=0)
use_date_filter = st.sidebar.checkbox("date_final로 기간 필터 사용", value=False)

filtered_df = base_df.copy()

if type_choice != "전체":
    filtered_df = filtered_df[filtered_df[COL_TYPE] == type_choice]

if use_date_filter:
    date_df = filtered_df.dropna(subset=[COL_DATE]).copy()
    if date_df.empty:
        st.sidebar.warning("선택 조건에서 date_final 값이 없어 기간 필터를 적용할 수 없습니다.")
    else:
        min_date = date_df[COL_DATE].min().date()
        max_date = date_df[COL_DATE].max().date()
        start_date, end_date = st.sidebar.date_input(
            "기간 선택 (date_final 기준)",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
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

# -----------------------------
# KPI
# -----------------------------
st.subheader("핵심 지표 (필터 반영)")
avg_comments = filtered_df[COL_COMMENTS].mean()
avg_likes = filtered_df[COL_LIKES].mean()
n_videos = len(filtered_df)

c1, c2, c3 = st.columns(3)
c1.metric("평균 댓글 참여율", f"{avg_comments:.6f}")
c2.metric("평균 좋아요 참여율", f"{avg_likes:.6f}")
c3.metric("영상 개수", f"{n_videos:,}")

with st.expander("📌 기초 통계 (전체/타입별)"):
    st.markdown("**전체 요약 통계**")
    st.dataframe(filtered_df[[COL_DURATION, COL_COMMENTS, COL_LIKES]].describe().round(6))

    st.markdown("**타입별 평균/중앙값/표준편차/개수**")
    grp = (
        filtered_df.groupby("type_label")[[COL_DURATION, COL_COMMENTS, COL_LIKES]]
        .agg(["mean", "median", "std", "count"])
        .round(6)
    )
    st.dataframe(grp)

# -----------------------------
# 상위 10% 영상 클릭 재생 기능
# -----------------------------
st.divider()
st.subheader("🎬 상위 10% 영상 보기 (클릭하면 앱에서 재생)")

metric_options = {
    "댓글 참여율 (Comments_Engagement)": COL_COMMENTS,
    "좋아요 참여율 (Likes_Engagement)": COL_LIKES,
}

if COL_VIEWS in filtered_df.columns:
    metric_options["조회수 (viewCount)"] = COL_VIEWS

metric_label = st.selectbox("기준 지표 선택", list(metric_options.keys()))
metric_col = metric_options[metric_label]

# 상위 10% 기준
metric_series = pd.to_numeric(filtered_df[metric_col], errors="coerce").dropna()
if metric_series.empty:
    st.info("선택한 지표에 유효한 값이 없어 상위 10%를 계산할 수 없습니다.")
else:
    threshold = metric_series.quantile(0.90)
    top_df = filtered_df[pd.to_numeric(filtered_df[metric_col], errors="coerce") >= threshold].copy()
    top_df = top_df.sort_values(metric_col, ascending=False)

    st.caption(f"상위 10% 기준값(90퍼센타일): **{threshold:.6f}** (지표: {metric_label})")
    st.write(f"상위 10% 영상 수: **{len(top_df)}개**")

    # 재생 상태
    if "selected_video_url" not in st.session_state:
        st.session_state.selected_video_url = None
        st.session_state.selected_video_title = None

    # 상위 10% 중 상위 N개만 버튼으로 보여주기 (너 요청: '상위 10% 안에 드는 영상' → UI 과밀 방지 위해 최대 30개)
    show_n = min(30, len(top_df))
    st.markdown(f"**상위 목록(최대 {show_n}개 표시)** — 버튼을 누르면 아래에서 재생됩니다.")

    for idx, row in top_df.head(show_n).iterrows():
        title = row[COL_TITLE] if COL_TITLE in top_df.columns else f"video_{idx}"
        vt = row["type_label"]
        dt = row["date_only"]
        val = row[metric_col]
        url = row[COL_URL]

        cols = st.columns([6, 2, 2])
        with cols[0]:
            st.markdown(f"**{title}**  \n- 타입: {vt} | 날짜: {dt} | {metric_label}: `{val:.6f}`")
            # 링크도 같이 제공
            st.link_button("원본 링크 열기", url)
        with cols[2]:
            if st.button("▶ 보기", key=f"play_{metric_col}_{idx}"):
                st.session_state.selected_video_url = url
                st.session_state.selected_video_title = title

        st.divider()

    if st.session_state.selected_video_url:
        st.markdown(f"### ▶ 재생 중: {st.session_state.selected_video_title}")
        st.video(st.session_state.selected_video_url)

# -----------------------------
# 시각화 탭
# -----------------------------
st.divider()
st.subheader("시각화")

tab1, tab2, tab3, tab4 = st.tabs(["분포/길이", "타입 비교", "시간 추이", "회차 기준 쌍비교(±1일)"])

# 1) 분포/길이
with tab1:
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered_df,
            x=COL_COMMENTS,
            color="type_label" if type_choice == "전체" else None,
            nbins=30,
            title="댓글 참여율 분포 (히스토그램)",
            labels={COL_COMMENTS: "댓글 참여율", "type_label": "타입"},
        )
        fig.update_layout(font=PLOTLY_FONT, legend_title_text="타입")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("해석: 분포가 낮은 구간에 몰리면 대부분 콘텐츠의 댓글 참여율이 낮고, 일부만 높을 가능성이 큽니다.")

    with right:
        fig = px.scatter(
            filtered_df,
            x=COL_DURATION,
            y=COL_COMMENTS,
            color="type_label" if type_choice == "전체" else None,
            title="영상 길이(duration_final) vs 댓글 참여율 (산점도)",
            labels={COL_DURATION: "영상 길이(초)", COL_COMMENTS: "댓글 참여율", "type_label": "타입"},
            hover_data=[COL_LIKES] + ([COL_VIEWS] if COL_VIEWS in filtered_df.columns else [])
        )
        fig.update_layout(font=PLOTLY_FONT, legend_title_text="타입")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("해석: duration_final 구간별로 댓글 참여율이 달라지는지, 포맷 간 패턴이 다른지 관찰합니다.")

# 2) 타입 비교
with tab2:
    colA, colB = st.columns(2)

    with colA:
        fig = px.box(
            filtered_df,
            x="type_label",
            y=COL_COMMENTS,
            points="all",
            title="타입별 댓글 참여율 분포 (Box + 점)",
            labels={"type_label": "타입", COL_COMMENTS: "댓글 참여율"},
        )
        fig.update_layout(font=PLOTLY_FONT)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.box(
            filtered_df,
            x="type_label",
            y=COL_LIKES,
            points="all",
            title="타입별 좋아요 참여율 분포 (Box + 점)",
            labels={"type_label": "타입", COL_LIKES: "좋아요 참여율"},
        )
        fig.update_layout(font=PLOTLY_FONT)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("해석: 중앙값/분산/이상치를 통해 숏폼과 롱폼의 반응 구조 차이를 비교합니다.")

# 3) 시간 추이
with tab3:
    time_df = filtered_df.dropna(subset=[COL_DATE]).copy()
    if time_df.empty:
        st.info("현재 필터 조건에서는 date_final 값이 없어 시간 추이를 그릴 수 없습니다.")
    else:
        daily = (
            time_df.groupby([time_df[COL_DATE].dt.date, "type_label"])[[COL_COMMENTS, COL_LIKES]]
            .mean()
            .reset_index()
        )
        daily = daily.rename(columns={daily.columns[0]: "date"})

        fig = px.line(
            daily,
            x="date",
            y=COL_COMMENTS,
            color="type_label",
            title="날짜별 평균 댓글 참여율",
            labels={"date": "날짜", COL_COMMENTS: "평균 댓글 참여율", "type_label": "타입"},
        )
        fig.update_layout(font=PLOTLY_FONT, legend_title_text="타입")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            daily,
            x="date",
            y=COL_LIKES,
            color="type_label",
            title="날짜별 평균 좋아요 참여율",
            labels={"date": "날짜", COL_LIKES: "평균 좋아요 참여율", "type_label": "타입"},
        )
        fig2.update_layout(font=PLOTLY_FONT, legend_title_text="타입")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("해석: 업로드 타이밍에 따라 참여율이 출렁이는지, 포맷별 추이가 다른지 확인합니다.")

# 4) 회차 기준 쌍비교(±1일)
with tab4:
    st.markdown("### 회차 기준 쌍비교 로직")
    st.markdown("- 같은 날짜에 올라온 롱폼/숏폼, 또는 업로드 날짜가 **±1일 이내**인 롱폼/숏폼은 같은 회차(같은 클립)로 간주합니다.")
    st.markdown("- 각 롱폼에 대해 조건(±1일)을 만족하는 숏폼이 여러 개면, **업로드 날짜 차이가 가장 작은** 숏폼 1개를 매칭합니다.")

    # 매칭 함수: 각 long에 대해 shorts 후보(±1일) 중 가장 가까운 1개 선택
    long_df = filtered_df[filtered_df[COL_TYPE] == "long"].dropna(subset=[COL_DATE]).copy()
    shorts_df = filtered_df[filtered_df[COL_TYPE] == "shorts"].dropna(subset=[COL_DATE]).copy()

    if long_df.empty or shorts_df.empty:
        st.info("현재 필터 조건에서 롱폼/숏폼 둘 다 존재해야 쌍비교가 가능합니다. (필터를 '전체'로 두는 것을 추천)")
    else:
        long_df = long_df.reset_index(drop=False).rename(columns={"index": "long_idx"})
        shorts_df = shorts_df.reset_index(drop=False).rename(columns={"index": "short_idx"})

        # 모든 조합 만들기(규모가 크지 않다는 전제) + ±1일 조건 필터
        pairs = long_df.assign(key=1).merge(shorts_df.assign(key=1), on="key", suffixes=("_long", "_short")).drop(columns=["key"])
        pairs["day_diff"] = (pairs[f"{COL_DATE}_long"] - pairs[f"{COL_DATE}_short"]).abs().dt.days
        pairs = pairs[pairs["day_diff"] <= 1].copy()

        if pairs.empty:
            st.warning("±1일 조건을 만족하는 롱폼-숏폼 쌍이 없습니다. date_final 값 또는 필터 범위를 확인해 주세요.")
        else:
            # 롱폼별로 day_diff 최소인 숏폼 1개 선택
            pairs = pairs.sort_values(["long_idx", "day_diff"])
            best_pairs = pairs.groupby("long_idx").head(1).copy()

            # 쌍 데이터 프레임 구성
            out_cols = {
                f"{COL_TITLE}_long": "롱폼 제목",
                f"{COL_URL}_long": "롱폼 URL",
                f"{COL_TITLE}_short": "숏폼 제목",
                f"{COL_URL}_short": "숏폼 URL",
                f"{COL_DATE}_long": "롱폼 날짜",
                f"{COL_DATE}_short": "숏폼 날짜",
                "day_diff": "날짜 차이(일)",
                f"{COL_COMMENTS}_long": "롱폼 댓글참여율",
                f"{COL_COMMENTS}_short": "숏폼 댓글참여율",
                f"{COL_LIKES}_long": "롱폼 좋아요참여율",
                f"{COL_LIKES}_short": "숏폼 좋아요참여율",
            }
            pair_df = best_pairs[list(out_cols.keys())].rename(columns=out_cols)

            # 차이(롱 - 숏)
            pair_df["댓글참여율 차이(롱-숏)"] = pair_df["롱폼 댓글참여율"] - pair_df["숏폼 댓글참여율"]
            pair_df["좋아요참여율 차이(롱-숏)"] = pair_df["롱폼 좋아요참여율"] - pair_df["숏폼 좋아요참여율"]

            st.write(f"매칭된 쌍 개수: **{len(pair_df)}쌍**")

            # 시각화: paired scatter (롱 vs 숏)
            left, right = st.columns(2)

            with left:
                fig = px.scatter(
                    pair_df,
                    x="숏폼 댓글참여율",
                    y="롱폼 댓글참여율",
                    title="쌍비교: 숏폼 vs 롱폼 댓글 참여율 (각 점=같은 회차 쌍)",
                    labels={"숏폼 댓글참여율": "숏폼 댓글참여율", "롱폼 댓글참여율": "롱폼 댓글참여율"},
                    hover_data=["롱폼 제목", "숏폼 제목", "날짜 차이(일)"]
                )
                fig.update_layout(font=PLOTLY_FONT)
                st.plotly_chart(fig, use_container_width=True)

                st.caption("해석: 대각선(y=x) 위에 점이 많으면 같은 회차에서 롱폼 댓글 참여율이 더 높은 경향을 의미합니다.")

            with right:
                fig = px.histogram(
                    pair_df,
                    x="댓글참여율 차이(롱-숏)",
                    nbins=30,
                    title="쌍비교: 댓글 참여율 차이 분포 (롱폼 - 숏폼)",
                    labels={"댓글참여율 차이(롱-숏)": "댓글 참여율 차이(롱-숏)"},
                )
                fig.update_layout(font=PLOTLY_FONT)
                st.plotly_chart(fig, use_container_width=True)

                st.caption("해석: 0보다 큰 구간에 몰릴수록, 같은 회차에서 롱폼이 숏폼보다 댓글 참여율이 높았던 경우가 많음을 의미합니다.")

            # 요약 통계
            st.markdown("### 쌍비교 요약 통계")
            d_comment = pair_df["댓글참여율 차이(롱-숏)"]
            d_like = pair_df["좋아요참여율 차이(롱-숏)"]

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("댓글 차이 평균(롱-숏)", f"{d_comment.mean():.6f}")
            s2.metric("댓글 차이 중앙값(롱-숏)", f"{d_comment.median():.6f}")
            s3.metric("좋아요 차이 평균(롱-숏)", f"{d_like.mean():.6f}")
            s4.metric("좋아요 차이 중앙값(롱-숏)", f"{d_like.median():.6f}")

            with st.expander("쌍 목록 보기 (URL 클릭 가능)"):
                # URL을 클릭 가능한 링크로 보이게
                show = pair_df.copy()
                show["롱폼 URL"] = show["롱폼 URL"].apply(lambda u: f"[링크]({u})")
                show["숏폼 URL"] = show["숏폼 URL"].apply(lambda u: f"[링크]({u})")
                st.markdown(show.to_markdown(index=False), unsafe_allow_html=True)

            # 쌍 선택해서 바로 재생
            st.markdown("### 🎞️ 쌍 선택 재생")
            pair_df = pair_df.reset_index(drop=True)
            pick = st.selectbox("재생할 쌍 선택", options=list(range(len(pair_df))), format_func=lambda i: f"{i+1}. {pair_df.loc[i,'롱폼 제목']}")
            st.write("롱폼")
            st.video(pair_df.loc[pick, "롱폼 URL"])
            st.write("숏폼")
            st.video(pair_df.loc[pick, "숏폼 URL"])

# -----------------------------
# 데이터 미리보기
# -----------------------------
with st.expander("필터 적용 데이터 보기"):
    preview_cols = [c for c in [COL_DATE, COL_TYPE, COL_DURATION, COL_COMMENTS, COL_LIKES, COL_VIEWS, COL_URL, COL_TITLE] if c in filtered_df.columns]
    st.dataframe(filtered_df[preview_cols].reset_index(drop=True))
