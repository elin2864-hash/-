# app.py
# 환승연애4 숏폼 vs 롱폼 참여 분석 Streamlit 앱 (배포/한글/채팅/확장 시각화 버전)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="환승연애4 | 숏폼 vs 롱폼 참여 분석", layout="wide")

st.title("환승연애4 유튜브 숏폼 vs 롱폼 참여 분석 대시보드")
st.caption("동일 클립 기반 콘텐츠라도 포맷(type)에 따라 참여(댓글/좋아요) 양상이 달라지는지 탐색합니다.")

# -----------------------------
# 너 데이터셋 '최종' 컬럼명 고정
# -----------------------------
COL_TYPE = "type"
COL_DURATION = "duration_final"
COL_DATE = "date_final"
COL_COMMENTS = "Comments_Engagement"
COL_LIKES = "Likes_Engagement"

REQUIRED_COLS = [COL_TYPE, COL_DURATION, COL_DATE, COL_COMMENTS, COL_LIKES]
DATA_PATH = "2237001_강선우_최종 데이터셋.xlsx"

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
df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

base_df = df.dropna(subset=[COL_TYPE, COL_DURATION, COL_COMMENTS, COL_LIKES]).copy()
if base_df.empty:
    st.warning("분석에 필요한 핵심 값 결측이 많아 시각화할 데이터가 없습니다.")
    st.stop()

# type 라벨(범례용) 명시
type_label_map = {"long": "롱폼(long)", "shorts": "숏폼(shorts)"}
base_df["type_label"] = base_df[COL_TYPE].map(type_label_map).fillna(base_df[COL_TYPE])

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

# Plotly 폰트(브라우저 폰트 우선)
PLOTLY_FONT = dict(family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif", size=14)

# -----------------------------
# KPI / 기초 통계
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

    st.markdown("**타입별 평균/중앙값/표준편차**")
    grp = (
        filtered_df.groupby("type_label")[[COL_DURATION, COL_COMMENTS, COL_LIKES]]
        .agg(["mean", "median", "std", "count"])
    )
    st.dataframe(grp.round(6))

# -----------------------------
# 시각화 영역
# -----------------------------
st.subheader("시각화")

tab1, tab2, tab3, tab4 = st.tabs(["분포/길이", "타입 비교", "시간 추이", "관계/상관"])

# 1) 분포 & 산점도
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

        st.caption("해석: 분포가 낮은 구간에 몰리면 대부분 콘텐츠의 댓글 참여율이 낮고, 일부만 높을 가능성이 큽니다. 포맷별 분포 차이를 함께 확인하세요.")

    with right:
        fig = px.scatter(
            filtered_df,
            x=COL_DURATION,
            y=COL_COMMENTS,
            color="type_label" if type_choice == "전체" else None,
            title="영상 길이(duration_final) vs 댓글 참여율 (산점도)",
            labels={COL_DURATION: "영상 길이(초)", COL_COMMENTS: "댓글 참여율", "type_label": "타입"},
            hover_data=[COL_LIKES]
        )
        fig.update_layout(font=PLOTLY_FONT, legend_title_text="타입")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("해석: duration_final 구간에 따라 댓글 참여율이 달라지는지, 그리고 포맷 간 패턴이 다른지 관찰합니다.")

# 2) 타입 비교(박스플롯 등)
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

    st.caption("해석: 중앙값과 분산(상자 폭), 이상치(튀는 점)를 통해 숏폼/롱폼의 반응 구조 차이를 비교합니다.")

# 3) 날짜별 추이
with tab3:
    st.markdown("**date_final 기준 평균 참여율 추이** (데이터가 충분할 때 의미가 큼)")

    time_df = filtered_df.dropna(subset=[COL_DATE]).copy()
    if time_df.empty:
        st.info("현재 필터 조건에서는 date_final 값이 없어 시간 추이를 그릴 수 없습니다.")
    else:
        daily = (
            time_df.groupby([time_df[COL_DATE].dt.date, "type_label"])[[COL_COMMENTS, COL_LIKES]]
            .mean()
            .reset_index()
            .rename(columns={COL_DATE: "date"})
        )
        daily = daily.rename(columns={daily.columns[0]: "date"})  # 안전

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

# 4) 관계/상관
with tab4:
    colC, colD = st.columns(2)

    with colC:
        fig = px.scatter(
            filtered_df,
            x=COL_LIKES,
            y=COL_COMMENTS,
            color="type_label" if type_choice == "전체" else None,
            title="좋아요 참여율 vs 댓글 참여율 (관계)",
            labels={COL_LIKES: "좋아요 참여율", COL_COMMENTS: "댓글 참여율", "type_label": "타입"},
        )
        fig.update_layout(font=PLOTLY_FONT, legend_title_text="타입")
        st.plotly_chart(fig, use_container_width=True)

    with colD:
        # 상관 히트맵
        corr_cols = [COL_DURATION, COL_COMMENTS, COL_LIKES]
        corr = filtered_df[corr_cols].corr(numeric_only=True)

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=["영상 길이", "댓글 참여율", "좋아요 참여율"],
            y=["영상 길이", "댓글 참여율", "좋아요 참여율"],
            zmin=-1, zmax=1,
            hoverongaps=False
        ))
        fig.update_layout(title="상관관계 히트맵", font=PLOTLY_FONT)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("해석: 좋아요와 댓글이 함께 움직이는지, 길이와 참여가 연관되는지 ‘방향성’ 중심으로 확인합니다(인과 단정 X).")

# -----------------------------
# 상위 콘텐츠 / 데이터 보기
# -----------------------------
with st.expander("🏆 상위 콘텐츠(참여율 기준) & 데이터 보기"):
    # 상위 15개(댓글 참여율)
    top_n = 15
    top_comments = filtered_df.sort_values(COL_COMMENTS, ascending=False).head(top_n)
    st.markdown(f"**댓글 참여율 TOP {top_n}**")
    show_cols = [COL_DATE, COL_TYPE, COL_DURATION, COL_COMMENTS, COL_LIKES]
    existing_cols = [c for c in show_cols if c in filtered_df.columns]
    st.dataframe(top_comments[existing_cols].reset_index(drop=True))

    st.markdown("**필터 적용 데이터 미리보기**")
    st.dataframe(filtered_df[existing_cols].reset_index(drop=True))

# -----------------------------
# 채팅형 해석 인터페이스
# -----------------------------
st.divider()
st.subheader("💬 결과 해석 도우미 (채팅형)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "질문을 입력하면, 현재 필터 기준 분석 결과로 답해줄게. 예) '숏폼이랑 롱폼 댓글 참여율 차이 요약해줘' / '길이랑 댓글 관계 어때?'"}
    ]

def summarize_now(df_now: pd.DataFrame) -> str:
    # 타입별 요약
    g = df_now.groupby("type_label")[[COL_COMMENTS, COL_LIKES, COL_DURATION]].mean()
    g = g.rename(columns={COL_COMMENTS: "avg_comments", COL_LIKES: "avg_likes", COL_DURATION: "avg_duration"})
    parts = []
    for idx in g.index:
        parts.append(
            f"- {idx}: 평균 댓글 {g.loc[idx,'avg_comments']:.6f}, 평균 좋아요 {g.loc[idx,'avg_likes']:.6f}, 평균 길이 {g.loc[idx,'avg_duration']:.1f}s"
        )
    # 상관
    corr = df_now[[COL_DURATION, COL_COMMENTS, COL_LIKES]].corr(numeric_only=True)
    cd = corr.loc[COL_DURATION, COL_COMMENTS]
    return "\n".join(parts) + f"\n\n- 길이-댓글 상관(피어슨): {cd:.3f} (방향성 참고, 인과 아님)"

def simple_chat_answer(user_text: str, df_now: pd.DataFrame) -> str:
    t = user_text.lower()

    # 공통 요약
    summary = summarize_now(df_now)

    if any(k in t for k in ["요약", "정리", "전체", "한줄", "결론"]):
        return f"현재 필터 기준 요약이야:\n{summary}"

    if any(k in t for k in ["숏", "short", "shorts", "롱", "long", "차이", "비교"]):
        # 차이 계산
        grp = df_now.groupby("type_label")[[COL_COMMENTS, COL_LIKES]].mean()
        if "숏폼(shorts)" in grp.index and "롱폼(long)" in grp.index:
            diff_c = grp.loc["롱폼(long)", COL_COMMENTS] - grp.loc["숏폼(shorts)", COL_COMMENTS]
            diff_l = grp.loc["롱폼(long)", COL_LIKES] - grp.loc["숏폼(shorts)", COL_LIKES]
            direction_c = "롱폼이 더 높음" if diff_c > 0 else "숏폼이 더 높음"
            direction_l = "롱폼이 더 높음" if diff_l > 0 else "숏폼이 더 높음"
            return (
                f"포맷 비교 결과(평균 기준):\n"
                f"- 댓글 참여율 차이(롱폼-숏폼): {diff_c:.6f} → {direction_c}\n"
                f"- 좋아요 참여율 차이(롱폼-숏폼): {diff_l:.6f} → {direction_l}\n\n"
                f"참고로 전체 요약:\n{summary}"
            )
        return f"현재 필터에서 숏폼/롱폼이 모두 포함되지 않아 직접 비교가 어려워. (필터를 '전체'로 두고 다시 질문해줘)\n\n{summary}"

    if any(k in t for k in ["길이", "duration", "상관", "관계"]):
        corr = df_now[[COL_DURATION, COL_COMMENTS, COL_LIKES]].corr(numeric_only=True)
        cd = corr.loc[COL_DURATION, COL_COMMENTS]
        ld = corr.loc[COL_DURATION, COL_LIKES]
        return (
            f"현재 필터에서 길이와 참여 지표의 관계(피어슨 상관)야:\n"
            f"- 길이 vs 댓글: {cd:.3f}\n"
            f"- 길이 vs 좋아요: {ld:.3f}\n\n"
            f"상관은 '같이 움직이는 방향'만 보여주고, 인과는 아니야.\n\n{summary}"
        )

    if any(k in t for k in ["좋아요", "likes", "댓글", "comments"]):
        return f"현재 필터 기준 참여 지표 요약:\n{summary}"

    # 기본 응답
    return f"질문을 조금 더 구체적으로 써주면(비교/관계/결론 등) 그 포인트로 계산해서 답해줄게.\n\n현재 필터 요약:\n{summary}"

# 출력
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("예: '숏폼 vs 롱폼 댓글 참여율 차이 요약해줘'")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    answer = simple_chat_answer(prompt, filtered_df)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
