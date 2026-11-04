import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson

# 1. Cấu hình trang web App hiển thị rộng giao diện
st.set_page_config(page_title="La Liga xG Predictor", layout="wide", page_icon="⚽")

# 2. Đọc dữ liệu sạch bằng cơ chế Cache để tối ưu hóa tốc độ tải trang
@st.cache_data
def load_data():
    # Đọc file dữ liệu sạch định dạng CSV
    df = pd.read_csv("matches_laliga_clean.csv")
    return df

try:
    df = load_data()

    # Tính toán các chỉ số bàn thắng kỳ vọng (xG) trung bình của toàn giải đấu
    avg_home_xg = df[df['venue'] == 'Home']['xg'].mean()
    avg_away_xg = df[df['venue'] == 'Away']['xg'].mean()

    # 3. Thiết kế thanh Menu điều hướng bên hông (Sidebar)
    st.sidebar.title("Menu Tính Năng")
    page = st.sidebar.selectbox("Chọn chức năng hệ thống", ["Match Prediction", "Team Performance Analysis"])

    # CHỨC NĂNG 1: DỰ ĐOÁN TRẬN ĐẤU (MATCH PREDICTION)
    if page == "Match Prediction":
        st.title("🏟️ Hệ Thống Dự Đoán Trận Đấu & Chỉ Số xG")
        st.markdown("Hệ thống áp dụng mô hình phân phối Poisson dựa trên 4.700 dòng dữ liệu lịch sử giải La Liga.")

        # Hộp chọn đội bóng song song
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Chọn Đội Chủ Nhà (Home)", sorted(df['team'].unique()))
        with col2:
            away_team = st.selectbox("Chọn Đội Khách (Away)", sorted(df['team'].unique()))

        # Ràng buộc logic không cho chọn trùng đội
        if home_team == away_team:
            st.warning("⚠️ Vui lòng chọn hai đội bóng khác nhau để thực hiện dự đoán toán học.")
        else:
            # Tính toán Sức mạnh Tấn công (Attack) và Phòng ngự (Defense) của Đội chủ nhà
            home_att = df[(df['team'] == home_team) & (df['venue'] == 'Home')]['xg'].mean() / avg_home_xg
            home_def = df[(df['team'] == home_team) & (df['venue'] == 'Home')]['xga'].mean() / avg_away_xg

            # Tính toán Sức mạnh Tấn công (Attack) và Phòng ngự (Defense) của Đội khách
            away_att = df[(df['team'] == away_team) & (df['venue'] == 'Away')]['xg'].mean() / avg_away_xg
            away_def = df[(df['team'] == away_team) & (df['venue'] == 'Away')]['xga'].mean() / avg_home_xg

            # Tính điểm số xG kỳ vọng phân phối cho trận đấu cụ thể này
            expected_home_xg = home_att * away_def * avg_home_xg
            expected_away_xg = away_att * home_def * avg_away_xg

            # Hiển thị kết quả Card số liệu trực quan
            st.subheader("📊 Điểm số Bàn thắng Kỳ vọng (xG) dự báo")
            c1, c2 = st.columns(2)
            c1.metric(label=f"xG Kỳ vọng: {home_team} (Home)", value=f"{expected_home_xg:.2f}")
            c2.metric(label=f"xG Kỳ vọng: {away_team} (Away)", value=f"{expected_away_xg:.2f}")

            # Thuật toán tính ma trận xác suất Thắng - Hòa - Thua (Phân phối Poisson)
            max_goals = 6
            home_probs = [poisson.pmf(i, expected_home_xg) for i in range(max_goals)]
            away_probs = [poisson.pmf(i, expected_away_xg) for i in range(max_goals)]

            prob_matrix = np.outer(home_probs, away_probs)
            prob_home_win = np.sum(np.tril(prob_matrix, -1))
            prob_draw = np.sum(np.diag(prob_matrix))
            prob_away_win = np.sum(np.triu(prob_matrix, 1))

            # Trực quan hóa xác suất kết quả bằng biểu đồ tròn (Pie Chart) của Plotly
            st.subheader("🎯 Xác Suất Kết Quả Trận Đấu")
            prob_df = pd.DataFrame({
                'Kết quả trận đấu': [f"{home_team} Thắng", 'Hòa Trận', f"{away_team} Thắng"],
                'Tỷ lệ (%)': [prob_home_win * 100, prob_draw * 100, prob_away_win * 100]
            })
            fig_pie = px.pie(prob_df, values='Tỷ lệ (%)', names='Kết quả trận đấu', 
                             color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
            st.plotly_chart(fig_pie, use_container_width=True)

    # CHỨC NĂNG 2: PHÂN TÍCH HIỆU SUẤT ĐỘI BÓNG (TEAM ANALYSIS)
    else:
        st.title("📈 Biểu Đồ Xu Hướng Phong Độ Đội Bóng Qua Các Mùa Giải")
        selected_team = st.selectbox("Chọn tên đội bóng cần phân tích", sorted(df['team'].unique()))

        team_df = df[df['team'] == selected_team]
        
        # Biểu đồ đường xu hướng giá trị ghi bàn (xG) và thủng lưới (xGA) qua từng mùa giải
        trend_df = team_df.groupby('season')[['xg', 'xga']].mean().reset_index()
        fig_line = px.line(trend_df, x='season', y=['xg', 'xga'], 
                           labels={'value': 'Chỉ số trung bình', 'season': 'Mùa giải'},
                           title=f"Xu hướng hiệu suất Tấn công (xG) và Phòng ngự (xGA) của câu lạc bộ {selected_team}")
        st.plotly_chart(fig_line, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Không tìm thấy file dữ liệu sạch 'matches_laliga_clean.csv'. Vui lòng kiểm tra lại thư mục dự án.")
