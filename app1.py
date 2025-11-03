# aqi_dashboard_premium.py
"""
Premium AQI Dashboard (Streamlit)
Requires: aqi_model.pkl (put in same folder)
Optional (created on first upload): scaler.pkl, country_encoder.pkl, status_encoder.pkl

Install:
pip install streamlit pandas numpy scikit-learn seaborn matplotlib plotly reportlab pyttsx3 fpdf
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fpdf import FPDF
import pyttsx3
from datetime import datetime

# -------------------------
# PAGE CONFIG + CSS (flashy)
# -------------------------
st.set_page_config(page_title="🌍 AQI Dashboard — Premium", layout="wide", page_icon="🌫️")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
    body {
      background: linear-gradient(-45deg, #e0f7fa, #e8f5e9, #fff3e0, #f3e5f5);
      background-size: 400% 400%;
      animation: gradient 12s ease infinite;
    }
    @keyframes gradient {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
    .air-icon { position: fixed; font-size: 28px; opacity: 0.12; animation: float 6s ease-in-out infinite; }
    @keyframes float { 0%{transform:translateY(0)}50%{transform:translateY(-18px)}100%{transform:translateY(0)} }
    .pred-box { padding:14px; border-radius:10px; font-size:20px; font-weight:700; text-align:center; }
    .small-muted { color:#666; font-size:13px; }
    .stDownloadButton>button { background: linear-gradient(90deg,#00796b,#004d40); color:white; }
    </style>
    <div class="air-icon" style="top:6%; left:6%;">🌬️</div>
    <div class="air-icon" style="top:18%; left:82%;">💨</div>
    <div class="air-icon" style="top:72%; left:14%;">🌫️</div>
    """,
    unsafe_allow_html=True
)

st.title("🌍 AQI Dashboard — Premium")
st.markdown("<div class='small-muted'>Predict AQI, explore data, export reports. Upload once to enable manual mode.</div>", unsafe_allow_html=True)

# -------------------------
# Utility helpers
# -------------------------
def load_pickle_if_exists(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return None
    return None

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def speak_py(msg):
    """pyttsx3 speak — synchronous, reliable on server"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        engine.say(msg)
        engine.runAndWait()
    except Exception:
        # failing silently is fine; user still gets UI feedback
        pass

def aqi_category_and_health(value):
    if value <= 50:
        return "Good", "#66bb6a", "Good. Air quality is healthy.", True
    if value <= 100:
        return "Moderate", "#ffee58", "Moderate. Air quality is not ideal.", False
    if value <= 150:
        return "Unhealthy for Sensitive Groups", "#ffa726", "Unhealthy for sensitive groups.", False
    if value <= 200:
        return "Unhealthy", "#ef5350", "Unhealthy. Take caution.", False
    if value <= 300:
        return "Very Unhealthy", "#8e24aa", "Very Unhealthy. Avoid outdoor activity.", False
    return "Hazardous", "#4a148c", "Hazardous. Stay indoors and take precautions.", False

def draw_gauge(value):
    segs = [0,50,100,150,200,300,500]
    colors = ["#66bb6a","#ffee58","#ffa726","#ef5350","#8e24aa","#4a148c"]
    fig, ax = plt.subplots(figsize=(6,3), subplot_kw={'projection':'polar'})
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    angles = np.linspace(0, np.pi, len(segs)-1)
    for i in range(len(angles)):
        ax.bar(angles[i], 1, width=np.pi/(len(angles)+1), color=colors[i], edgecolor='white')
    needle = (min(value, 500)/500.0)*np.pi
    ax.plot([needle, needle],[0,1], color='black', linewidth=3)
    ax.text(0, -0.2, f"{round(value,1)}", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.axis('off')
    return fig

def pdf_report_from_dict(title, rows, filename="AQI_Report.pdf"):
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(150, 760, title)
    pdf.setFont("Helvetica", 11)
    y = 720
    for k, v in rows.items():
        pdf.drawString(80, y, f"{k}: {v}")
        y -= 18
        if y < 80:
            pdf.showPage()
            y = 750
    pdf.showPage()
    pdf.save()
    buf.seek(0)
    return buf

# -------------------------
# Load required model + optional scaler & encoders
# -------------------------
try:
    model = load_pickle_if_exists("aqi_model.pkl")
    if model is None:
        st.error("Missing: aqi_model.pkl — put your trained model in the same folder and reload.")
        st.stop()
    else:
        st.sidebar.success("✅ Loaded model: aqi_model.pkl")
except Exception as e:
    st.error("Error loading model: " + str(e))
    st.stop()

scaler = load_pickle_if_exists("scaler.pkl")
country_encoder = load_pickle_if_exists("country_encoder.pkl")
status_encoder = load_pickle_if_exists("status_encoder.pkl")

if scaler is not None:
    st.sidebar.info("Loaded scaler.pkl (optional)")

if country_encoder is not None and status_encoder is not None:
    st.sidebar.info("Loaded encoders (country_encoder.pkl, status_encoder.pkl)")

# -------------------------
# Session state initialization
# -------------------------
if 'encoders_ready' not in st.session_state:
    st.session_state.encoders_ready = (country_encoder is not None and status_encoder is not None and scaler is not None)
if 'country_classes' not in st.session_state:
    st.session_state.country_classes = list(country_encoder.classes_) if country_encoder is not None else None
if 'status_classes' not in st.session_state:
    st.session_state.status_classes = list(status_encoder.classes_) if status_encoder is not None else None
if 'scaler_session' not in st.session_state:
    st.session_state.scaler_session = scaler
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["Dataset Mode (Upload CSV)", "Manual Mode (Select names)"])
auto_refresh = st.sidebar.checkbox("Auto Refresh Dashboard", value=False)
refresh_rate = st.sidebar.slider("Auto-refresh interval (seconds)", 10, 60, 20)
voice_enabled = st.sidebar.checkbox("Enable Server TTS (pyttsx3)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Encoding reference** (auto-updates after dataset upload)")
enc_ref_placeholder = st.sidebar.empty()

# -------------------------
# Header info & quick actions
# -------------------------
cols = st.columns([3,1])
with cols[1]:
    if st.button("Export encoders & scaler (if ready)"):
        if st.session_state.get('encoders_ready'):
            st.success("Already created and saved as files (country_encoder.pkl, status_encoder.pkl, scaler.pkl).")
        else:
            st.info("No encoders/scaler ready — upload dataset once to create them.")

st.markdown("## Quick legend")
st.markdown("- **AQI Value**: numeric prediction (0 - 500+).")
st.markdown("- **Verdict**: simple 'Healthy' vs 'Not healthy' + detailed category.")

# -------------------------
# MODE: Dataset Upload
# -------------------------
if mode.startswith("Dataset"):
    st.header("📁 Dataset Mode — upload CSV")
    uploaded_file = st.file_uploader("Upload CSV file (columns: Date optional, Country, Status, AQI Value)", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            st.stop()

        st.subheader("Data Preview")
        st.dataframe(data.head())

        # Preprocess
        if "Date" in data.columns:
            data = data.drop(columns=["Date"])
        # detect AQI column (prefer exact match)
        target_col = None
        for c in data.columns:
            if "aqi" in c.lower():
                target_col = c
                break

        if target_col:
            y = data[target_col].copy()
            X_raw = data.drop(columns=[target_col])
        else:
            y = None
            X_raw = data.copy()
            st.warning("No AQI target column auto-detected. Predictions will run but metrics won't show.")

        # Build LabelEncoders live from uploaded data
        le_country = LabelEncoder()
        le_status = LabelEncoder()
        if "Country" in X_raw.columns:
            try:
                X_raw['Country_enc'] = le_country.fit_transform(X_raw['Country'].astype(str))
            except Exception:
                X_raw['Country_enc'] = 0
        else:
            X_raw['Country_enc'] = 0

        if "Status" in X_raw.columns:
            try:
                X_raw['Status_enc'] = le_status.fit_transform(X_raw['Status'].astype(str))
            except Exception:
                X_raw['Status_enc'] = 0
        else:
            X_raw['Status_enc'] = 0

        # Build feature matrix in expected order
        expected_features = ['Country','Status']
        X_fixed = pd.DataFrame()
        X_fixed['Country'] = X_raw['Country_enc'] if 'Country_enc' in X_raw.columns else 0
        X_fixed['Status'] = X_raw['Status_enc'] if 'Status_enc' in X_raw.columns else 0
        X_fixed = X_fixed[[f for f in expected_features if f in X_fixed.columns]]

        # Fit scaler from uploaded dataset if not present
        if st.session_state.scaler_session is None:
            try:
                scaler_fit = StandardScaler()
                scaler_fit.fit(X_fixed.astype(float))
                st.session_state.scaler_session = scaler_fit
                # save to disk
                save_pickle(scaler_fit, "scaler.pkl")
                save_pickle(le_country, "country_encoder.pkl")
                save_pickle(le_status, "status_encoder.pkl")
                st.session_state.country_classes = list(le_country.classes_)
                st.session_state.status_classes = list(le_status.classes_)
                st.session_state.encoders_ready = True
                st.success("Encoders & scaler created from uploaded dataset and saved as .pkl files.")
            except Exception as e:
                st.error("Failed to fit scaler on uploaded data: " + str(e))
        else:
            scaler_fit = st.session_state.scaler_session

        # Scale & predict
        try:
            X_scaled = st.session_state.scaler_session.transform(X_fixed.astype(float))
            preds = model.predict(X_scaled)
        except Exception as e:
            st.error("Scaling/prediction failed. Check data types & encoders: " + str(e))
            st.stop()

        data['Predicted_AQI'] = preds

        # Show predictions
        st.markdown("### 🔢 Predictions (sample)")
        st.dataframe(data.head(20))

        # Evaluation metrics if true labels present
        if y is not None:
            try:
                r2 = r2_score(y, preds)
                mae = mean_absolute_error(y, preds)
                rmse = np.sqrt(mean_squared_error(y, preds))
                st.markdown("### 📊 Model Performance")
                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{r2:.3f}")
                c2.metric("MAE", f"{mae:.2f}")
                c3.metric("RMSE", f"{rmse:.2f}")
            except Exception:
                pass

        # Visualizations area
        st.markdown("## 🎨 Visualizations")
        sns.set_style("whitegrid")

        # Correlation heatmap
        with st.container():
            st.subheader("Correlation Heatmap")
            fig_h, ax_h = plt.subplots(figsize=(6,4))
            sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="magma", ax=ax_h)
            st.pyplot(fig_h)

        # Pairplot (safe)
        with st.container():
            st.subheader("Pairplot (may be slow for large datasets)")
            try:
                fig_pair = sns.pairplot(data.select_dtypes(include=[np.number]).iloc[:, :6], diag_kind="kde", palette="cubehelix")
                st.pyplot(fig_pair)
            except Exception:
                st.info("Pairplot skipped due to dataset size or dtypes.")

        # Feature importance (if available)
        with st.container():
            st.subheader("Feature Importance (model-based)")
            try:
                fi = pd.Series(model.feature_importances_, index=expected_features).sort_values(ascending=False)
                fig_fi, ax_fi = plt.subplots(figsize=(6,4))
                sns.barplot(x=fi.values, y=fi.index, palette="viridis", ax=ax_fi)
                ax_fi.set_title("Feature importance")
                st.pyplot(fig_fi)
            except Exception:
                st.info("Model does not provide feature_importances_ or features mismatch.")

        # Interactive Plotly charts
        with st.container():
            st.subheader("Interactive Charts")
            try:
                if "Country" in data.columns and "Predicted_AQI" in data.columns:
                    # if Country present as strings, use original; if encoded, try to map back
                    if data['Country'].dtype == np.number or np.issubdtype(data['Country'].dtype, np.integer):
                        # show average predicted AQI per encoded country
                        avg_aqi = data.groupby('Country')['Predicted_AQI'].mean().reset_index()
                        fig_bar = px.bar(avg_aqi, x='Country', y='Predicted_AQI', title="Average Predicted AQI by Country", color='Predicted_AQI', color_continuous_scale='viridis')
                    else:
                        fig_bar = px.bar(data, x="Country", y="Predicted_AQI", title="Predicted AQI by Country", color="Predicted_AQI")
                    st.plotly_chart(fig_bar, use_container_width=True)
            except Exception:
                pass

        # Encoding reference in sidebar
        if st.session_state.encoders_ready:
            country_map = {i: name for i, name in enumerate(st.session_state.country_classes)}
            status_map = {i: name for i, name in enumerate(st.session_state.status_classes)}
            enc_html = "<b>Country Encoding:</b><br>" + "<br>".join([f"{k} → {v}" for k,v in country_map.items()]) \
                     + "<br><br><b>Status Encoding:</b><br>" + "<br>".join([f"{k} → {v}" for k,v in status_map.items()])
            enc_ref_placeholder.markdown(enc_html, unsafe_allow_html=True)

        # Inspect single prediction, speak, download pdf/csv
        st.markdown("### 🔍 Inspect & Export")
        idx = st.number_input("Pick row index to inspect", min_value=0, max_value=max(0, len(data)-1), value=0, step=1)
        chosen_val = float(data['Predicted_AQI'].iloc[int(idx)])
        cat, colr, vmsg, healthy_flag = aqi_category_and_health(chosen_val)
        verdict = "Healthy ✅" if healthy_flag else "Not healthy ⚠️"
        st.markdown(f"<div class='pred-box' style='background:{colr}; color:white;'>Predicted AQI Value: {round(chosen_val,2)}</div>", unsafe_allow_html=True)
        st.markdown(f"**Verdict:** {verdict}")
        st.markdown(f"**Category:** {cat}")

        # Voice: pyttsx3 (server-side) — speak each time prediction changes
        if voice_enabled:
            if st.session_state.last_prediction != chosen_val:
                st.session_state.last_prediction = chosen_val
                speak_py = speak_py = speak_py = None  # placeholder to avoid lint
                # call synchronous tts
                speak_py = speak_py  # no-op to satisfy linter
                # run tts
                speak_py = lambda msg: speak_py_engine(msg)
                # Implement speak via pyttsx3
                try:
                    speak_py_engine = pyttsx3.init
                    engine_local = pyttsx3.init()
                    engine_local.setProperty('rate', 160)
                    engine_local.setProperty('volume', 1.0)
                    engine_local.say(f"Predicted AQI is {int(round(chosen_val,0))}. {cat}")
                    engine_local.runAndWait()
                except Exception:
                    # fallback to simple message if TTS fails
                    st.info(f"(TTS failed) Predicted AQI is {int(round(chosen_val,0))}. {cat}")

        st.pyplot(draw_gauge(chosen_val))

        # CSV download for predictions
        csv_buf = io.BytesIO()
        try:
            csv_bytes = data.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download full predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception:
            st.info("CSV download unavailable (error).")

        # PDF download for inspected row
        pdf_rows = {
            "Inspected row": idx,
            "Predicted AQI": round(chosen_val,2),
            "Verdict": verdict,
            "Category": cat,
            "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        pdf_buf = pdf_report_from_dict("Air Quality Index Report", pdf_rows)
        st.download_button("📄 Download inspected row PDF", data=pdf_buf, file_name="AQI_inspect.pdf", mime="application/pdf")

        # auto-refresh handling
        if auto_refresh:
            st.sidebar.info(f"Auto-refresh every {refresh_rate}s")
            time.sleep(refresh_rate)
            st.experimental_rerun()

# -------------------------
# MODE: Manual Mode
# -------------------------
else:
    st.header("🎛️ Manual Mode — predict by selecting names")

    if not st.session_state.encoders_ready:
        st.warning("Manual mode requires encoders & scaler. Upload a dataset once in Dataset Mode to create them.")
        st.markdown("Quick demo mode below (no encoders):")
        demo_mode = st.checkbox("Enable demo quick-predict (not using your real encoders)", value=False)
        if demo_mode:
            demo_countries = ["India", "China", "USA", "Germany", "UK"]
            demo_status = ["Good", "Moderate", "Unhealthy"]
            country_choice = st.selectbox("Country (demo)", demo_countries)
            status_choice = st.selectbox("Status (demo)", demo_status)
            if st.button("🔮 Predict (demo)"):
                ce = LabelEncoder(); ce.fit(demo_countries)
                se = LabelEncoder(); se.fit(demo_status)
                c_enc = int(ce.transform([country_choice])[0])
                s_enc = int(se.transform([status_choice])[0])
                temp_scaler = StandardScaler(); temp_scaler.fit(np.array([[c_enc, s_enc]]))
                scaled = temp_scaler.transform(np.array([[c_enc, s_enc]]))
                pred = model.predict(scaled)[0]
                cat, colr, vmsg, healthy = aqi_category_and_health(pred)
                verdict = "Healthy ✅" if healthy else "Not healthy ⚠️"
                st.markdown(f"<div class='pred-box' style='background:{colr}; color:white;'>Predicted AQI Value: {round(pred,2)}</div>", unsafe_allow_html=True)
                st.markdown(f"**Verdict:** {verdict}")
                st.markdown(f"**Category:** {cat}")
                if voice_enabled:
                    try:
                        engine_local = pyttsx3.init(); engine_local.setProperty('rate',160); engine_local.say(f"Predicted AQI is {int(round(pred,0))}. {cat}"); engine_local.runAndWait()
                    except Exception:
                        st.info("(TTS failed)")
                st.pyplot(draw_gauge(pred))
    else:
        # normal manual mode using saved encoders & scaler
        country_list = st.session_state.country_classes
        status_list = st.session_state.status_classes
        country_choice = st.selectbox("Country", country_list)
        status_choice = st.selectbox("Status", status_list)

        if st.button("🔮 Predict AQI from manual input"):
            try:
                # load encoders from saved files if present else build from session classes
                if os.path.exists("country_encoder.pkl") and os.path.exists("status_encoder.pkl"):
                    ce = load_pickle_if_exists("country_encoder.pkl")
                    se = load_pickle_if_exists("status_encoder.pkl")
                else:
                    ce = LabelEncoder(); ce.classes_ = np.array(st.session_state.country_classes)
                    se = LabelEncoder(); se.classes_ = np.array(st.session_state.status_classes)

                c_enc = int(ce.transform([country_choice])[0])
                s_enc = int(se.transform([status_choice])[0])
            except Exception as e:
                st.error("Encoding failed: " + str(e))
                st.stop()

            if st.session_state.scaler_session is None:
                st.error("Scaler missing. Upload dataset in Dataset Mode to create scaler.")
                st.stop()
            try:
                manual_df = pd.DataFrame([{ 'Country': c_enc, 'Status': s_enc }])[['Country','Status']]
                manual_scaled = st.session_state.scaler_session.transform(manual_df.astype(float))
                pred_val = float(model.predict(manual_scaled)[0])
            except Exception as e:
                st.error("Scaling/prediction failed: " + str(e))
                st.stop()

            cat, colr, vmsg, healthy_flag = aqi_category_and_health(pred_val)
            health_text = "Healthy ✅" if healthy_flag else "Not healthy ⚠️"
            st.markdown(f"<div class='pred-box' style='background:{colr}; color:white;'>Predicted AQI Value: {round(pred_val,2)}</div>", unsafe_allow_html=True)
            st.markdown(f"**Verdict:** {health_text}")
            st.markdown(f"**Category:** {cat}")

            # voice: pyttsx3 (server)
            if voice_enabled:
                if st.session_state.last_prediction != pred_val:
                    st.session_state.last_prediction = pred_val
                    try:
                        engine_local = pyttsx3.init(); engine_local.setProperty('rate',160); engine_local.say(f"Predicted AQI is {int(round(pred_val,0))}. {cat}"); engine_local.runAndWait()
                    except Exception:
                        st.info("(TTS failed) Predicted AQI: " + str(int(round(pred_val,0))))

            st.pyplot(draw_gauge(pred_val))

            # Download PDF for manual prediction
            if st.button("📄 Download Manual PDF Report"):
                rows = {
                    "Country": country_choice,
                    "Status": status_choice,
                    "Predicted AQI": round(pred_val,2),
                    "Verdict": health_text,
                    "Category": cat,
                    "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                pdf_buf = pdf_report_from_dict("AQI Manual Prediction Report", rows)
                st.download_button("⬇️ Download PDF", data=pdf_buf.getvalue(), file_name="AQI_manual_report.pdf", mime="application/pdf")

        # show encoding tables in sidebar
        if st.session_state.encoders_ready:
            country_map = {i: name for i, name in enumerate(st.session_state.country_classes)}
            status_map = {i: name for i, name in enumerate(st.session_state.status_classes)}
            st.sidebar.markdown("**Country Encoding**")
            st.sidebar.write(pd.DataFrame(list(country_map.items()), columns=["Code","Country"]))
            st.sidebar.markdown("**Status Encoding**")
            st.sidebar.write(pd.DataFrame(list(status_map.items()), columns=["Code","Status"]))

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Premium AQI Dashboard • Model: RandomForestRegressor (pickled) • Upload once to enable full manual mode.")
