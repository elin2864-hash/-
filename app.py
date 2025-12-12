import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from openai import OpenAI

# Font verification function
def setup_korean_font():
    # Windows (Local)
    if os.name == 'nt':
        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)
    # Linux (Streamlit Cloud)
    else:
        # Check if NanumGothic is installed or download it
        font_path = "NanumGothic.ttf"
        if not os.path.exists(font_path):
            import requests
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            response = requests.get(url)
            with open(font_path, "wb") as f:
                f.write(response.content)
        
        fm.fontManager.addfont(font_path)
        plt.rc('font', family='NanumGothic')
        plt.rc('axes', unicode_minus=False)

setup_korean_font()

# Page Config
st.set_page_config(
    page_title="서울시 상권 분석 AI 어시스턴트",
    page_icon="🏙️",
    layout="wide"
)

# Title
st.title("🏙️ 서울시 상권 분석 및 매출 예측 AI")
st.markdown("서울시 상권 데이터를 기반으로 매출을 예측하고, AI와 대화하며 인사이트를 얻어보세요.")

# Sidebar
st.sidebar.header("설정 (Settings)")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="결과 해석을 위해 API 키가 필요합니다.")

# Data Loading function
@st.cache_data
def load_data():
    try:
        # Load datasets
        df_pop = pd.read_csv("서울시 상권분석서비스(길단위인구-상권).csv", encoding='cp949')
        df_change = pd.read_csv("서울시 상권분석서비스(상권변화지표-상권).csv", encoding='cp949')
        # Handle files in data/ folder or current folder if migrated
        try:
            df_store = pd.read_csv("data/서울시 상권분석서비스(점포-상권)_2024년.csv", encoding='cp949')
        except FileNotFoundError:
             df_store = pd.read_csv("서울시 상권분석서비스(점포-상권)_2024년.csv", encoding='cp949')
        
        try:
            df_sales = pd.read_csv("data/서울시 상권분석서비스(추정매출-상권)_2024년.csv", encoding='cp949')
        except FileNotFoundError:
            df_sales = pd.read_csv("서울시 상권분석서비스(추정매출-상권)_2024년.csv", encoding='cp949')

        # Merge Data
        # Strategy: Merge basic info first.
        # Use inner join to keep only matching records across all datasets
        
        # 1. Pop + Change
        df_merged = pd.merge(df_pop, df_change, on=['기준_년분기_코드', '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명'], how='inner')
        
        # 2. Add Store info
        # Store data might have multiple rows per district (different service codes). 
        # For simplicity in this regression, let's aggregate store counts per district/quarter
        # Or better, filter for a specific service code if asked, but here we want general 'District' analysis.
        # Aggregating store metrics by district and quarter
        store_agg = df_store.groupby(['기준_년분기_코드', '상권_코드']).agg({
            '점포_수': 'sum',
            '프랜차이즈_점포_수': 'sum',
            '개업_점포_수': 'sum',
            '폐업_점포_수': 'sum'
        }).reset_index()
        
        df_merged = pd.merge(df_merged, store_agg, on=['기준_년분기_코드', '상권_코드'], how='inner')
        
        # 3. Add Sales info (Target)
        # Sales data also split by service code. We should aggregate total sales for the district for a holistic view
        # OR allow user to select service code.
        # Let's aggregate for now to predict "Total District Sales"
        sales_agg = df_sales.groupby(['기준_년분기_코드', '상권_코드']).agg({
            '당월_매출_금액': 'sum',
            '당월_매출_건수': 'sum',
            '주중_매출_금액': 'sum',
            '주말_매출_금액': 'sum'
        }).reset_index()
        
        df_merged = pd.merge(df_merged, sales_agg, on=['기준_년분기_코드', '상권_코드'], how='inner')
        
        return df_merged
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return None

df = load_data()

if df is not None:
    st.sidebar.success("데이터 로드 완료! (Row: " + str(len(df)) + ")")
    
    # ----------------------------------------
    # Filters
    # ----------------------------------------
    st.sidebar.subheader("데이터 필터")
    quarters = sorted(df['기준_년분기_코드'].unique())
    selected_quarter = st.sidebar.selectbox("분기 선택", quarters, index=len(quarters)-1)
    
    # Filter Data for Display/Analysis context (optional, maybe we want to train on ALL and predict/analyze specific)
    # Let's keep all data for training to get better model, but highlight selected data.
    
    # Selection of Features for Regression
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove ID cols and Target cols from features
    exclude_cols = ['기준_년분기_코드', '상권_구분_코드', '상권_코드', '당월_매출_금액', '당월_매출_건수', '주중_매출_금액', '주말_매출_금액']
    feature_candidates = [c for c in numeric_cols if c not in exclude_cols]
    
    # Default features
    default_features = ['총_유동인구_수', '점포_수', '프랜차이즈_점포_수', '운영_영업_개월_평균']
    default_features = [f for f in default_features if f in feature_candidates]
    
    selected_features = st.multiselect("학습 할 Feature 선택", feature_candidates, default=default_features)
    
    target_col = '당월_매출_금액'
    
    if st.button("분석 및 예측 실행 (Run Analysis)"):
        if not selected_features:
            st.warning("Feature를 최소 하나 이상 선택해주세요.")
            st.stop()
            
        # ----------------------------------------
        # Regression Analysis
        # ----------------------------------------
        X = df[selected_features]
        y = df[target_col]
        
        # Simple fillna just in case
        X = X.fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        st.divider()
        st.header("📊 회귀 분석 결과")
        
        col1, col2 = st.columns(2)
        col1.metric("R-Squared (결정계수)", f"{r2:.4f}")
        col2.metric("MSE (평균제곱오차)", f"{mse:,.0f}")
        
        # ----------------------------------------
        # Visualization
        # ----------------------------------------
        st.subheader("1. Feature Importance (회귀 계수)")
        coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_})
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
        
        fig_coef, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=coef_df, x='Coefficient', y='Feature', ax=ax, palette='viridis')
        ax.set_title("각 변수가 매출에 미치는 영향도")
        st.pyplot(fig_coef)
        
        st.subheader("2. Actual vs Predicted Sales")
        fig_scatter, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        # Ideal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_xlabel("실제 매출")
        ax.set_ylabel("예측 매출")
        ax.set_title("실제 매출 vs 예측 매출 산점도")
        st.pyplot(fig_scatter)
        
        # ----------------------------------------
        # Chat Interface Integration
        # ----------------------------------------
        st.divider()
        st.header("🤖 AI 분석 결과 해석")
        st.caption("위 분석 결과를 바탕으로 AI와 대화해보세요.")

        # Store analysis context in session state to pass to LLM
        analysis_summary = f"""
        **회귀 분석 요약**:
        - 타겟 변수: 상권 월 매출액
        - 사용 변수: {', '.join(selected_features)}
        - 모델 성능 (R2): {r2:.4f}
        
        **주요 변수 영향도 (계수)**:
        {coef_df.to_string(index=False)}
        """
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Initial system message
            st.session_state.messages.append({
                "role": "system", 
                "content": f"당신은 유능한 데이터 분석가입니다. 다음은 서울시 상권 분석 데이터에 대한 회귀분석 결과입니다. 사용자의 질문에 대해 결과를 바탕으로 인사이트를 제공하고 쉽게 설명해주세요.\n\n[분석 결과 데이터]\n{analysis_summary}"
            })
            # Add initial AI greeting
            st.session_state.messages.append({"role": "assistant", "content": "분석이 완료되었습니다! 결과에 대해 궁금한 점을 물어보세요."})

        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Chat Input
        if prompt := st.chat_input("이 결과가 무슨 의미인가요?"):
            if not api_key:
                st.error("OpenAI API Key를 입력해주세요.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Stream response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        client = OpenAI(api_key=api_key)
                        
                        # Call API
                        # Filter system message + last N messages to fit context if needed, but usually fine for simple chats
                        stream = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                        )
                        
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.write(full_response + "▌")
                        
                        message_placeholder.write(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
else:
    st.info("데이터를 불러오는 중입니다...")
