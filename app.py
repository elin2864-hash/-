import streamlit as st
import sys
import io

# Streamlit Cloud 배포 시 한글 인코딩 오류 해결
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from openai import OpenAI

# 페이지 설정
st.set_page_config(page_title="보스턴 집값 분석 & 챗봇", layout="wide")

# 제목
st.title("🏡 보스턴 집값 데이터 회귀 분석 및 AI 챗봇")

# 사이드바: OpenAI API 키 입력
st.sidebar.header("설정")
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['PRICE'] = target
    return df

# 메인 분석 로직
try:
    df = load_data()

    # 데이터 분할
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 분석 결과 요약 텍스트 생성 (챗봇 컨텍스트용)
    coefficients = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
    analysis_summary = f"""
    [분석 요약]
    - 모델: 선형 회귀 (Linear Regression)
    - MSE (평균 제곱 오차): {mse:.2f}
    - R2 Score (결정 계수): {r2:.2f}
    
    [주요 변수 영향도 (계수)]
    상위 3개 양의 상관관계:
    {coefficients.head(3).to_string()}
    
    상위 3개 음의 상관관계:
    {coefficients.tail(3).to_string()}
    """

    # 레이아웃: 2단 컬럼
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 데이터 분석 및 시각화")
        
        # 1. 상관관계 히트맵
        st.markdown("### 상관관계 히트맵")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

        # 2. 실제값 vs 예측값
        st.markdown("### 실제값 vs 예측값 (Test Set)")
        fig_scat, ax_scat = plt.subplots(figsize=(8, 6))
        ax_scat.scatter(y_test, y_pred, alpha=0.7)
        ax_scat.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax_scat.set_xlabel("Actual Price")
        ax_scat.set_ylabel("Predicted Price")
        ax_scat.set_title(f"R2 Score: {r2:.2f}")
        st.pyplot(fig_scat)

        st.info(analysis_summary)

    with col2:
        st.subheader("💬 AI 데이터 분석가")
        
        if not api_key:
            st.warning("사이드바에 OpenAI API 키를 입력해주세요.")
        else:
            # OpenAI 클라이언트 초기화
            client = OpenAI(api_key=api_key)

            # 세션 스테이트 초기화
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": f"당신은 데이터 분석 전문가입니다. 다음은 보스턴 집값 데이터의 회귀 분석 결과입니다. 이 결과를 바탕으로 사용자의 질문에 친절하고 전문적으로 답변해주세요.\n\n{analysis_summary}"}
                ]

            # 채팅 기록 표시
            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # 사용자 입력
            if prompt := st.chat_input("데이터에 대해 궁금한 점을 물어보세요!"):
                # 사용자 메시지 표시
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # AI 응답 생성 (스트리밍)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        stream = client.chat.completions.create(
                            model="gpt-4o-mini", # 또는 gpt-3.5-turbo, gpt-4 등
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                        )
                        
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        
                        # 응답 저장
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")

except Exception as e:
    st.error(f"데이터 로드 또는 분석 중 오류 발생: {e}")
