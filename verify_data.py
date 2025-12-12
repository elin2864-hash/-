import pandas as pd
import os

def load_data():
    try:
        print("Loading files...")
        # Load datasets
        df_pop = pd.read_csv("서울시 상권분석서비스(길단위인구-상권).csv", encoding='cp949')
        print(f"Pop loaded: {df_pop.shape}")
        
        df_change = pd.read_csv("서울시 상권분석서비스(상권변화지표-상권).csv", encoding='cp949')
        print(f"Change loaded: {df_change.shape}")
        
        # Handle files in data/ folder or current folder if migrated
        try:
            df_store = pd.read_csv("data/서울시 상권분석서비스(점포-상권)_2024년.csv", encoding='cp949')
        except FileNotFoundError:
             df_store = pd.read_csv("서울시 상권분석서비스(점포-상권)_2024년.csv", encoding='cp949')
        print(f"Store loaded: {df_store.shape}")
        
        try:
            df_sales = pd.read_csv("data/서울시 상권분석서비스(추정매출-상권)_2024년.csv", encoding='cp949')
        except FileNotFoundError:
            df_sales = pd.read_csv("서울시 상권분석서비스(추정매출-상권)_2024년.csv", encoding='cp949')
        print(f"Sales loaded: {df_sales.shape}")

        # Merge Data
        print("Merging...")
        
        # 1. Pop + Change
        df_merged = pd.merge(df_pop, df_change, on=['기준_년분기_코드', '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명'], how='inner')
        print(f"Merged 1 (Pop+Change): {df_merged.shape}")
        
        # 2. Add Store info
        store_agg = df_store.groupby(['기준_년분기_코드', '상권_코드']).agg({
            '점포_수': 'sum',
            '프랜차이즈_점포_수': 'sum',
            '개업_점포_수': 'sum',
            '폐업_점포_수': 'sum'
        }).reset_index()
        
        df_merged = pd.merge(df_merged, store_agg, on=['기준_년분기_코드', '상권_코드'], how='inner')
        print(f"Merged 2 (Store): {df_merged.shape}")
        
        # 3. Add Sales info (Target)
        sales_agg = df_sales.groupby(['기준_년분기_코드', '상권_코드']).agg({
            '당월_매출_금액': 'sum',
            '당월_매출_건수': 'sum',
            '주중_매출_금액': 'sum',
            '주말_매출_금액': 'sum'
        }).reset_index()
        
        df_merged = pd.merge(df_merged, sales_agg, on=['기준_년분기_코드', '상권_코드'], how='inner')
        print(f"Merged 3 (Sales): {df_merged.shape}")
        
        return df_merged
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    df = load_data()
    if df is not None and not df.empty:
        print("SUCCESS: Data loaded and merged.")
        print(df.head())
    else:
        print("FAILURE: Data not loaded correctly.")
