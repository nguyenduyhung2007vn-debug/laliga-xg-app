import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Kiểm tra tính sẵn sàng của thư viện XGBoost tăng cường hiệu suất
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# =====================================================
# CẤU HÌNH GIAO DIỆN PREMIUM CYBER DARK - CHỮ PHÁT SÁNG
# =====================================================
st.set_page_config(
    page_title="LaLiga xG Predictor & Analytics",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
/* 1. Đổi phông nền tổng thể sang màu đen sâu tăng độ tương phản chữ */
.stApp { 
    background-color: #060913 !important; 
} 

/* 2. ÉP TOÀN BỘ CHỮ VĂN BẢN VÀ CÁC THÀNH PHẦN CON SÁNG TRẮNG TUYỆT ĐỐI */
.stApp p, .stApp span, .stApp label, .stApp li, .stApp div, .stApp small {
    color: #ffffff !important;
    font-weight: 500 !important;
    text-shadow: 0px 0px 1px rgba(255,255,255,0.2) !important;
}

/* 3. Cấu hình riêng cho thanh Sidebar bên trái và chữ trong Sidebar */
[data-testid="stSidebar"] { 
    background-color: #0b0f19 !important; 
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
}

/* 4. Định dạng các ô Thẻ hiển thị bo góc (Card Layout) */
[data-testid="stMetric"], .stMetric {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    padding: 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6) !important;
}

/* Ép các con số KPI lớn phát sáng màu xanh dạ quang rực rỡ */
[data-testid="stMetricValue"] {
    color: #00ffcc !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.4) !important;
}

/* 5. ÉP CÁC THẺ TIÊU ĐỀ H1, H2, H3 PHÁT SÁNG MÀU XANH NEON CYBERPUNK */
h1, h2, h3, h1 *, h2 * , h3 * { 
    color: #00f0ff !important; 
    font-weight: 800 !important;
    text-shadow: 0px 0px 8px rgba(0, 240, 255, 0.5) !important;
    font-family: 'Inter', sans-serif !important;
} 

/* 6. Định dạng thanh chọn Tabs */
.stTabs [data-baseweb="tab"] { 
    color: #64748b !important; 
} 
.stTabs [data-baseweb="tab"][aria-selected="true"] { 
    color: #00f0ff !important; 
    font-weight: bold !important; 
    border-bottom-color: #00f0ff !important;
}

/* 7. Làm đẹp Hộp lựa chọn Selectbox và ép chữ bên trong luôn trắng sáng */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}
.stSelectbox div[data-baseweb="select"] * {
    color: #ffffff !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "matches_laliga_clean.csv"

# =====================================================
# TIỀN XỬ LÝ DỮ LIỆU & TẠO ĐẶC TRƯNG (FEATURE ENGINEERING)
# =====================================================
@st.cache_data
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    
    df["team"] = df["team"].astype(str).str.strip()
    df["opponent"] = df["opponent"].astype(str).str.strip()
    df["venue"] = df["venue"].astype(str).str.strip().str.lower().replace({
        "home": "Home", "h": "Home", "away": "Away", "a": "Away"
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    for col in ["xg", "xga", "gf", "ga", "sh", "sot"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df = df.dropna(subset=["team", "venue", "opponent", "xg", "xga", "gf", "ga", "date"]).copy()
    df = df[df["venue"].isin(["Home", "Away"])].sort_values("date").reset_index(drop=True)

    # --- TRÍCH XUẤT ĐẶC TRƯNG ---
    df["is_home"] = (df["venue"] == "Home").astype(int)
    
    df["xg_avg5"] = df.groupby("team")["xg"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["xga_avg5"] = df.groupby("team")["xga"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["goals_avg5"] = df.groupby("team")["gf"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["goals_against_avg5"] = df.groupby("team")["ga"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["sot_avg5"] = df.groupby("team")["sot"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["xg_avg3"] = df.groupby("team")["xg"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["net_xg_avg5"] = df["xg_avg5"] - df["xga_avg5"]
    
    df["matches_seen"] = df.groupby("team").cumcount()
    df["short_history"] = (df["matches_seen"] < 5).astype(int)
    
    opp_df = df[["date", "team", "opponent", "xg_avg5", "xga_avg5", "matches_seen", "short_history"]].copy()
    opp_df = opp_df.rename(columns={
        "team": "opponent", "opponent": "team",
        "xg_avg5": "opponent_xg_avg5", "xga_avg5": "opponent_xga_avg5",
        "matches_seen": "opponent_matches_seen", "short_history": "opponent_short_history"
    })
    
    final_df = pd.merge(df, opp_df, on=["date", "team", "opponent"], how="left")
    final_df["attack_defense_index"] = (final_df["xg_avg5"] + final_df["opponent_xga_avg5"]) / 2
    
    feature_cols = [
        "xg_avg5", "xga_avg5", "goals_avg5", "goals_against_avg5", "sot_avg5", "xg_avg3",
        "opponent_xg_avg5", "opponent_xga_avg5", "is_home", "matches_seen", "short_history",
        "opponent_matches_seen", "opponent_short_history", "net_xg_avg5", "attack_defense_index"
    ]
    
    train_end_idx = int(len(final_df) * 0.8)
    train_medians = final_df.iloc[:train_end_idx][feature_cols].median()
    
    for col in feature_cols:
        final_df[col] = final_df[col].fillna(train_medians[col])
        
    return final_df, feature_cols
# =====================================================
# HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING & BỘ DỰ BÁO POISSON
# =====================================================
@st.cache_resource
def train_ml_models(df, feature_cols):
    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df["xg"]
    X_test, y_test = test_df[feature_cols], test_df["xg"]
    
    # Mô hình 1: Random Forest
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Mô hình 2: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Mô hình 3: XGBoost
    xgb_model = None
    xgb_pred = None
    if HAS_XGBOOST:
        xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
    base_pred = X_test["xg_avg5"].values
    
    metrics = {
        "Random Forest": {
            "MAE": mean_absolute_error(y_test, rf_pred), 
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred))
        },
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test, lr_pred), 
            "RMSE": np.sqrt(mean_squared_error(y_test, lr_pred))
        },
        "Historical Baseline": {
            "MAE": mean_absolute_error(y_test, base_pred), 
            "RMSE": np.sqrt(mean_squared_error(y_test, base_pred))
        }
    }
    
    if HAS_XGBOOST:
        metrics["XGBoost"] = {
            "MAE": mean_absolute_error(y_test, xgb_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred))
        }
    
    best_model = rf_model
    best_pred = rf_pred
    best_model_name = "Random Forest"
    
    if HAS_XGBOOST and metrics["XGBoost"]["MAE"] < metrics["Random Forest"]["MAE"]:
        best_model = xgb_model
        best_pred = xgb_pred
        best_model_name = "XGBoost"
        
    return best_model, best_model_name, metrics, X_test, y_test, best_pred

def poisson_matrix(home_xg, away_xg):
    max_goals = 10
    goals = np.arange(max_goals + 1)
    home_prob = poisson.pmf(goals, home_xg)
    away_prob = poisson.pmf(goals, away_xg)
    matrix = np.outer(home_prob, away_prob)
    return matrix / matrix.sum()

def calculate_outcomes(matrix):
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total
st.sidebar.title("⚽ LaLiga xG Predictor")
st.sidebar.caption("Machine Learning Analytics Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MENU CHỨC NĂNG",
    [
        "🏠 Tổng quan hệ thống",
        "🏟️ Match Prediction (Học máy)",
        "📊 Phân tích xG & Phong độ",
        "🤖 Đánh giá Mô hình Học máy"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Đề tài Học Máy 2026")

# Khởi chạy kiểm tra và liên kết tiến trình hệ thống
try:
    if not DATA_FILE.exists():
        st.error(f"Không tìm thấy file dữ liệu '{DATA_FILE.name}'. Vui lòng tải file 'matches_laliga_clean.csv' lên thư mục.")
        st.stop()
        
    with st.spinner("⚽ Hệ thống đang huấn luyện AI và tính toán phong độ... Vui lòng đợi trong giây lát!"):
        df, feature_cols = load_and_process_data(DATA_FILE)
        best_model, best_model_name, model_metrics, X_test, y_test, test_preds = train_ml_models(df, feature_cols)
        teams = sorted(df["team"].unique().tolist())
    # ------------------ TRANG 1: TỔNG QUAN ------------------
    if page == "🏠 Tổng quan hệ thống":
        st.title("🏆 HỆ THỐNG PHÂN TÍCH BÓNG ĐÁ LALIGA")
        st.markdown(f"Mô hình thuật toán tối ưu hiện tại: **{best_model_name}**")
        
        # Sắp xếp 4 ô KPI dạng lưới mỏng nằm ngang giống ảnh mẫu
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng dữ liệu trận", f"{len(df):,}")
        c2.metric("Số CLB LaLiga", df["team"].nunique())
        c3.metric("Số mùa giải", df["season"].nunique())
        c4.metric("Bàn thắng TB/trận", f"{(df['gf'].mean() + df['ga'].mean())/2:.2f}")
        
        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📊 Mật độ phân phối xG (Home vs Away)")
            fig = px.histogram(df, x="xg", color="venue", barmode="overlay", marginal="box",
                               color_discrete_map={"Home": "#10b981", "Away": "#3b82f6"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_g2:
            st.subheader("📈 Ma trận tương quan các chỉ số tấn công")
            corr_matrix = df[["gf", "ga", "xg", "xga", "sh", "sot"]].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="Viridis")
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------ TRANG 2: MATCH PREDICTION ------------------
    elif page == "🏟️ Match Prediction (Học máy)":
        st.title("🏟️ DỰ ĐOÁN TRẬN ĐẤU SỬ DỤNG HỌC MÁY")
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("🏠 Chọn đội chủ nhà", teams, index=teams.index("Barcelona") if "Barcelona" in teams else 0)
        with col2:
            away_team = st.selectbox("🚀 Chọn đội khách", teams, index=teams.index("Real Madrid") if "Real Madrid" in teams else 1)
            
        if home_team == away_team:
            st.warning("Vui lòng chọn 2 câu lạc bộ khác nhau để thiết lập cặp đấu.")
        else:
            home_last_match = df[df["team"] == home_team].tail(1)
            away_last_match = df[df["team"] == away_team].tail(1)
            
            if home_last_match.empty or away_last_match.empty:
                st.error("Thiếu lịch sử phong độ để tạo đặc trưng cho cặp đấu này.")
            else:
                home_features = pd.DataFrame([{
                    "xg_avg5": home_last_match["xg_avg5"].values[0], "xga_avg5": home_last_match["xga_avg5"].values[0],
                    "goals_avg5": home_last_match["goals_avg5"].values[0], "goals_against_avg5": home_last_match["goals_against_avg5"].values[0],
                    "sot_avg5": home_last_match["sot_avg5"].values[0], "xg_avg3": home_last_match["xg_avg3"].values[0],
                    "opponent_xg_avg5": away_last_match["xg_avg5"].values[0], "opponent_xga_avg5": away_last_match["xga_avg5"].values[0],
                    "is_home": 1, "matches_seen": home_last_match["matches_seen"].values[0] + 1, "short_history": 0,
                    "opponent_matches_seen": away_last_match["matches_seen"].values[0] + 1, "opponent_short_history": 0,
                    "net_xg_avg5": home_last_match["net_xg_avg5"].values[0],
                    "attack_defense_index": (home_last_match["xg_avg5"].values[0] + away_last_match["xga_avg5"].values[0]) / 2
                }])
                
                away_features = pd.DataFrame([{
                    "xg_avg5": away_last_match["xg_avg5"].values[0], "xga_avg5": away_last_match["xga_avg5"].values[0],
                    "goals_avg5": away_last_match["goals_avg5"].values[0], "goals_against_avg5": away_last_match["goals_against_avg5"].values[0],
                    "sot_avg5": away_last_match["sot_avg5"].values[0], "xg_avg3": away_last_match["xg_avg3"].values[0],
                    "opponent_xg_avg5": home_last_match["xg_avg5"].values[0], "opponent_xga_avg5": home_last_match["xga_avg5"].values[0],
                    "is_home": 0, "matches_seen": away_last_match["matches_seen"].values[0] + 1, "short_history": 0,
                    "opponent_matches_seen": home_last_match["matches_seen"].values[0] + 1, "opponent_short_history": 0,
                    "net_xg_avg5": away_last_match["net_xg_avg5"].values[0],
                    "attack_defense_index": (away_last_match["xg_avg5"].values[0] + home_last_match["xga_avg5"].values[0]) / 2
                }])
                
                raw_pred_home = best_model.predict(home_features[feature_cols])
                raw_pred_away = best_model.predict(away_features[feature_cols])
                
                pred_home_xg = max(0.1, float(raw_pred_home[0] if isinstance(raw_pred_home, np.ndarray) and raw_pred_home.ndim > 0 else raw_pred_home))
                pred_away_xg = max(0.1, float(raw_pred_away[0] if isinstance(raw_pred_away, np.ndarray) and raw_pred_away.ndim > 0 else raw_pred_away))
                
                st.markdown("---")
                st.subheader("⚔️ KẾT QUẢ DỰ BÁO xG TỪ MÔ HÌNH HỌC MÁY")
                
                # Cấu trúc hiển thị mạng lưới chỉ số mỏng giống như ảnh Fixtures mẫu (Góc dưới bên trái ảnh mẫu)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(f"Dự báo xG {home_team}", f"{pred_home_xg:.2f}")
                k2.metric(f"Dự báo xG {away_team}", f"{pred_away_xg:.2f}")
                k3.metric("Lịch sử sút chính xác", f"{home_last_match['sot_avg5'].values[0]:.1f}")
                k4.metric("Chỉ số Công/Thủ kết hợp", f"{home_features['attack_defense_index'].values[0]:.2f}")
                
                matrix = poisson_matrix(pred_home_xg, pred_away_xg)
                home_win, draw, away_win = calculate_outcomes(matrix)
                
                # Biến đổi thành biểu đồ Thanh nằm ngang (Horizontal Bar Chart) cực kỳ sang trọng giống góc dưới ảnh mẫu
                st.markdown("---")
                st.subheader("🎯 XÁC SUẤT KẾT QUẢ TRẬN ĐẤU (Mô phỏng Poisson)")
                
                fig_prob = go.Figure(go.Bar(
                    y=[away_team, "Hòa", home_team],
                    x=[away_win * 100, draw * 100, home_win * 100],
                    orientation='h',
                    marker_color=["#3b82f6", "#fbbf24", "#10b981"],
                    text=[f"{away_win * 100:.1f}%", f"{draw * 100:.1f}%", f"{home_win * 100:.1f}%"],
                    textposition="inside"
                ))
                fig_prob.update_layout(
                    xaxis_title="Xác suất phần trăm (%)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_prob, use_container_width=True)

    # ------------------ TRANG 3: TEAM ANALYTICS ------------------
    elif page == "📊 Phân tích xG & Phong độ":
        st.title("📈 PHÂN TÍCH HỒ SƠ PHONG ĐỘ ĐỘI BÓNG")
        selected_team = st.selectbox("Chọn câu lạc bộ cần tra cứu", teams)
        team_df = df[df["team"] == selected_team].sort_values("date", ascending=False)
        
        st.subheader(f"Lịch sử thống kê hiệu suất thực tế của {selected_team}")
        st.dataframe(team_df[["date", "opponent", "venue", "gf", "ga", "xg", "xga", "season"]].head(15), use_container_width=True)
        
        st.markdown("---")
        st.subheader("⏳ Biểu đồ tiến trình xG & xGA trung bình theo mùa giải")
        seasonal_trend = team_df.groupby("season")[["xg", "xga"]].mean().reset_index()
        fig_trend = px.line(seasonal_trend, x="season", y=["xg", "xga"], markers=True,
                            title=f"Sự dịch chuyển chất lượng chiến thuật của {selected_team}")
        fig_trend.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_trend, use_container_width=True)
    # ------------------ TRANG 4: MODEL EVALUATION ------------------
    elif page == "🤖 Đánh giá Mô hình Học máy":
        st.title("🤖 THỰC NGHIỆM VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH")
        st.markdown("Báo cáo kiểm thử định lượng kết quả trên tập Test độc lập.")
        
        st.subheader("📊 Bảng so sánh sai số giữa các thuật toán")
        metrics_data = []
        for model_name, score in model_metrics.items():
            metrics_data.append({
                "Thuật toán mô hình": model_name,
                "MAE (Sai số tuyệt đối TB)": f"{score['MAE']:.4f}",
                "RMSE (Sai số bình phương TB)": f"{score['RMSE']:.4f}"
            })
        st.table(pd.DataFrame(metrics_data))
        
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("🎯 Mức độ quan trọng Đặc trưng")
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
                indices = np.argsort(importances)
                fig_importance = go.Figure(go.Bar(
                    x=[feature_cols[i] for i in indices], y=[importances[i] for i in indices],
                    orientation='v', marker_color="#38bdf8"
                ))
                fig_importance.update_layout(title=f"Mức độ đóng góp đặc trưng của {best_model_name}", template="plotly_dark")
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info(f"Mô hình tuyến tính ({best_model_name}) sử dụng hệ số Coefficients, không hỗ trợ hiển thị Feature Importance dạng cây.")
            
        with col_g2:
            st.subheader("📉 Phân tích Phần dư Sai số (Residuals)")
            residuals = y_test - test_preds
            fig_res = px.histogram(x=residuals, nbins=30, labels={'x': 'Sai số (Thực tế - Dự báo)'},
                                   title="Phân phối phần dư sai lệch của mô hình tốt nhất", color_discrete_sequence=["#ef4444"])
            fig_res.update_layout(template="plotly_dark")
            st.plotly_chart(fig_res, use_container_width=True)

except Exception as e:
    st.error(f"Ứng dụng gặp sự cố kỹ thuật trong quá trình xử lý: {e}")
    st.info("Hãy đảm bảo tệp dữ liệu 'matches_laliga_clean.csv' nằm cùng thư mục gốc chứa file python và có cấu trúc hợp lệ.")
