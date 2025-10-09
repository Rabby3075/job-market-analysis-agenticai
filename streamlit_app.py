"""
Streamlit Frontend for Job Market Analysis
Provides an interactive interface for the agentic AI system
"""

import glob
import io
import json
import os
import shutil
import time
import warnings
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

from agents.data_visualizer import DataVisualizer

# Page configuration
st.set_page_config(
    page_title="AU Job Market Analysis using Agentic AI",
    page_icon="🇦🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS to make Plotly charts stretch to full width and beautiful global styles
st.markdown("""
<style>
    .stPlotlyChart {
        width: 100% !important;
    }
    
    /* Global page styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Beautiful back button */
    .back-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin-bottom: 2rem;
    }
    
    .back-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Dashboard header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .dashboard-header h1 {
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Silence noisy FutureWarnings from plotly/pandas datetime conversions
warnings.filterwarnings("ignore", category=FutureWarning, module="_plotly_utils.basevalidators")
warnings.filterwarnings("ignore", message=".*DatetimeProperties.to_pydatetime is deprecated.*")
warnings.filterwarnings("ignore", message=".*The keyword arguments have been deprecated.*")

def check_backend_status():
    """Check if backend server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def auto_download_abs_latest():
    """Automatically download and process latest ABS data - INDUSTRY ONLY"""
    try:
        # Check backend status first
        if not check_backend_status():
            st.error("❌ **Backend server is not running!**\n\nPlease start the backend server first:\n1. Open a new terminal\n2. Run: `python main.py`\n3. Wait for 'Application startup complete'\n4. Then try again")
            return False
        
        with st.spinner("📥 Downloading latest ABS dataset (Industry only)..."):
            r = requests.post("http://localhost:8000/abs/download", json={"query": "latest"}, timeout=300)
            if r.status_code != 200:
                st.error(f"Download failed: {r.text}")
                return False
            
            download_info = r.json()
            paths = download_info.get("downloaded", [])
            if not paths:
                st.warning("No files downloaded.")
                return False
            
            # Filter for industry files only (Table 4)
            industry_paths = []
            for path in paths:
                try:
                    # Check the Index sheet to identify Table 4 (Industry) files
                    import pandas as pd
                    df_index = pd.read_excel(path, sheet_name='Index', header=None)
                    index_text = " ".join(df_index.fillna("").astype(str).values.ravel()).lower()
                    if "table 4" in index_text and "industry" in index_text:
                        industry_paths.append(path)
                except Exception as e:
                    # If we can't read the file, skip it
                    continue
            
            if not industry_paths:
                st.warning("No industry data found in downloaded files.")
                return False
            
        with st.spinner("📊 Processing industry data..."):
            pr = requests.post("http://localhost:8000/process-files", json={"paths": industry_paths}, timeout=600)
            if pr.status_code != 200:
                st.error(f"Processing failed: {pr.text}")
                return False
            
            pdata = pr.json()
            names = pdata.get("processed", [])
            if names:
                st.session_state.datasets = {name.get("dataset_name"): {"name": name.get("dataset_name")} for name in names}
                st.session_state.current_dataset = names[0].get("dataset_name")
            
        return True
        
    except requests.exceptions.ConnectionError:
        st.error("❌ **Backend server connection error!**\n\nPlease restart the backend server and try again.")
        return False
    except Exception as e:
        st.error(f"Auto-download error: {e}")
        return False

def auto_download_and_process_ivi():
    """Automatically download and process IVI data"""
    try:
        # Check backend status first
        if not check_backend_status():
            st.error("❌ **Backend server is not running!**\n\nPlease start the backend server first:\n1. Open a new terminal\n2. Run: `python main.py`\n3. Wait for 'Application startup complete'\n4. Then try again")
            return False
        
        with st.spinner("📥 Downloading latest IVI dataset..."):
            r = requests.post("http://localhost:8000/ivi/download", json={"file_type": "anzsco4_states"}, timeout=300)
            if r.status_code != 200:
                st.error(f"Download failed: {r.text}")
                return False
            
            download_info = r.json()
            paths = download_info.get("downloaded", [])
            if not paths:
                st.warning("No files downloaded.")
                return False
            
        with st.spinner("📊 Processing IVI data..."):
            # Process the downloaded files using the existing preprocessing pipeline
            pr = requests.post("http://localhost:8000/process-files", json={"paths": paths}, timeout=600)
            if pr.status_code != 200:
                st.error(f"Processing failed: {pr.text}")
                return False
            
            pdata = pr.json()
            names = pdata.get("processed", [])
            if names:
                st.session_state.datasets = {name.get("dataset_name"): {"name": name.get("dataset_name")} for name in names}
                st.session_state.current_dataset = "IVI IT Jobs Dataset"
            
        st.success("✅ IVI data downloaded and processed successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        st.error("❌ **Backend server connection error!**\n\nPlease restart the backend server and try again.")
        return False
    except Exception as e:
        st.error(f"IVI processing error: {e}")
        return False

def reset_all_data():
    """Reset all downloaded datasets and clear cache"""
    try:
        # Clear all CSV files in the current directory
        csv_files = glob.glob("*.csv")
        for file in csv_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        
        # Clear charts directory
        if os.path.exists("charts"):
            try:
                shutil.rmtree("charts")
                print("Deleted charts directory")
            except Exception as e:
                print(f"Error deleting charts directory: {e}")
        
        # Clear any other data directories
        data_dirs = ["data", "downloads", "processed"]
        for dir_name in data_dirs:
            if os.path.exists(dir_name):
                try:
                    shutil.rmtree(dir_name)
                    print(f"Deleted {dir_name} directory")
                except Exception as e:
                    print(f"Error deleting {dir_name} directory: {e}")
        
        # Clear Streamlit cache
        st.cache_data.clear()
        
        # Show success message
        st.success("✅ All data has been reset successfully!")
        
    except Exception as e:
        st.error(f"❌ Error resetting data: {str(e)}")
        print(f"Reset error: {e}")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None

# Clear invalid current_dataset if it doesn't exist in datasets
if (st.session_state.current_dataset and 
    st.session_state.datasets and 
    st.session_state.current_dataset not in st.session_state.datasets):
    st.session_state.current_dataset = None

# Ensure datasets is always a dictionary
if not isinstance(st.session_state.datasets, dict):
    st.session_state.datasets = {}

def main():
    """Main application function"""
    
    # Initialize session state for current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'landing'
    
    # Route to appropriate page
    if st.session_state.current_page == 'landing':
        show_landing_page()
    elif st.session_state.current_page == 'abs_dashboard':
        show_abs_dashboard()
    elif st.session_state.current_page == 'ivi_dashboard':
        show_ivi_dashboard()

def show_landing_page():
    """Display the landing page with title, intro, and navigation buttons"""
    
    # Add beautiful landing page styles
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Landing header with gradient background */
    .landing-header {
        text-align: center;
        margin-bottom: 4rem;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .landing-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .landing-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
    }
    
    .landing-header .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        font-weight: 300;
        position: relative;
        z-index: 1;
        margin-top: 1rem;
    }
    
    /* Intro section with glassmorphism effect */
    .intro-section {
        text-align: center;
        margin-bottom: 4rem;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .intro-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        border-radius: 20px 20px 0 0;
    }
    
    .intro-section p {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: #2d3748;
        margin: 0;
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Dashboard buttons section */
    .dashboard-section {
        margin-top: 3rem;
    }
    
    .dashboard-section h3 {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .dashboard-section h3::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Modern dashboard cards */
    .dashboard-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        border: 2px solid transparent;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 20px 20px 0 0;
    }
    
    .dashboard-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
    }
    
    .card-desc {
        font-family: 'Inter', sans-serif;
        color: #718096;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    
    /* Dashboard button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        margin-top: 1rem !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    }
    
    /* Reset button styling */
    .stButton button[kind="secondary"] {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3) !important;
        margin-top: 0 !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
        background: linear-gradient(135deg, #ff5252, #d32f2f) !important;
    }
    
    /* Feature highlights */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-top: 4rem;
        padding: 2rem;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-family: 'Inter', sans-serif;
        color: #718096;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .landing-header h1 {
            font-size: 2.5rem;
        }
        
        .dashboard-button-container {
            flex-direction: column;
            align-items: center;
        }
        
        .dashboard-button {
            min-width: 100%;
            max-width: 300px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title / Banner with subtitle
    st.markdown("""
    <div class="landing-header">
        <h1>🇦🇺 AU Job Market Analysis using Agentic AI</h1>
        <div class="subtitle">Powered by Advanced AI Agents • Real-time Data Analysis • Predictive Insights</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Short intro text with glassmorphism
    st.markdown("""
    <div class="intro-section">
        <p>Explore Australian IT job market trends from ABS & IVI datasets.<br>
        Use natural language queries or dashboards to analyze demand over time, including COVID impacts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered Analysis</div>
            <div class="feature-desc">Advanced machine learning algorithms provide deep insights into job market trends</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Interactive Dashboards</div>
            <div class="feature-desc">Beautiful, responsive visualizations that adapt to your data exploration needs</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <div class="feature-title">Predictive Forecasting</div>
            <div class="feature-desc">5-year predictions with confidence intervals for strategic planning</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard selection section with modern cards
    st.markdown("""
    <div class="dashboard-section">
        <h3>Choose Your Analysis Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern dashboard cards with proper buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-icon">📊</div>
            <div class="card-title">ABS Industry Dashboard</div>
            <div class="card-desc">Analyze Australian job market data by industry from ABS sources</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 ABS Industry Dashboard", key="abs_dashboard", use_container_width=True):
            # Automatically download and process latest ABS industry data
            if auto_download_abs_latest():
                st.session_state.current_page = "abs_dashboard"
                st.success("✅ Latest ABS industry data downloaded and processed successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to download ABS industry data. Please check backend connection.")
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-icon">💻</div>
            <div class="card-title">IVI (IT Jobs) Dashboard</div>
            <div class="card-desc">IT job market analysis from IVI datasets</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("💻 IVI (IT Jobs) Dashboard", key="ivi_dashboard", use_container_width=True):
            # Automatically download and process IVI data
            if auto_download_and_process_ivi():
                st.session_state.current_page = "ivi_dashboard"
                st.rerun()
            else:
                st.error("❌ Failed to download and process IVI data. Please check backend connection.")
    

def show_abs_dashboard():
    """Display the ABS Dataset Dashboard (existing functionality)"""
    
    # Add navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Home", key="back_from_abs"):
            st.session_state.current_page = "landing"
            st.rerun()
    
    with col3:
        if st.button("🗑️ Reset All Data", key="reset_data", type="secondary"):
            reset_all_data()
            st.session_state.current_page = "landing"
            st.rerun()
    
    # Beautiful dashboard header
    st.markdown("""
    <div class="dashboard-header">
        <h1>📊 ABS Industry Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern navigation styling for ABS dashboard
    st.markdown("""
    <style>
    .nav-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        justify-content: center;
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Modern sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    .css-1d391kg .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    .css-1d391kg .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Main content area styling */
    .main .block-container {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    
    /* Welcome section styling */
    .welcome-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .welcome-section h1 {
        color: #667eea;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .welcome-section p {
        color: #e2e8f0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .capabilities-list {
        text-align: left;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .capabilities-list li {
        color: #cbd5e0;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation buttons for ABS dashboard
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="nav-container">', unsafe_allow_html=True)
        
        # Check current ABS page
        if 'abs_current_page' not in st.session_state:
            st.session_state.abs_current_page = 'analysis'
        
        # Analysis page button
        if st.button("📊 Data Analysis", key="nav_analysis", 
                    type="primary" if st.session_state.abs_current_page == 'analysis' else "secondary"):
            st.session_state.abs_current_page = 'analysis'
            st.rerun()
        
        # Visualization page button  
        if st.button("📈 Visualizations", key="nav_viz",
                    type="primary" if st.session_state.abs_current_page == 'visualizations' else "secondary"):
            st.session_state.abs_current_page = 'visualizations'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Route to appropriate ABS page
    if st.session_state.abs_current_page == 'analysis':
        show_analysis_page()
    elif st.session_state.abs_current_page == 'visualizations':
        show_visualizations_page()

def show_ivi_dashboard():
    """Display the IVI Dashboard (same as ABS dashboard)"""
    
    # Add navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Home", key="back_from_ivi"):
            st.session_state.current_page = "landing"
            st.rerun()
    
    with col3:
        if st.button("🗑️ Reset All Data", key="reset_data_ivi", type="secondary"):
            reset_all_data()
            st.session_state.current_page = "landing"
            st.rerun()
    
    # Beautiful dashboard header
    st.markdown("""
    <div class="dashboard-header">
        <h1>💻 IVI (IT Jobs) Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern navigation styling for IVI dashboard
    st.markdown("""
    <style>
    .nav-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    .nav-button {
        flex: 1;
        padding: 1rem;
        border: 2px solid #667eea;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .nav-button.active {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border-color: #ff6b6b;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        if st.button("📊 Data Analysis", key="ivi_analysis", type="primary"):
            st.session_state.ivi_current_page = "analysis"
            st.rerun()
    
    with col_nav2:
        if st.button("📈 Visualizations", key="ivi_viz"):
            st.session_state.ivi_current_page = "visualizations"
            st.rerun()
    
    # Set default page and ensure proper dataset name
    if 'ivi_current_page' not in st.session_state:
        st.session_state.ivi_current_page = 'analysis'
    
    # Ensure IVI dataset name is set
    if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None or 'ivi' not in str(st.session_state.current_dataset).lower():
        st.session_state.current_dataset = 'IVI IT Jobs Dataset'
    
    # Show appropriate page
    if st.session_state.ivi_current_page == 'analysis':
        show_ivi_analysis_page()
    elif st.session_state.ivi_current_page == 'visualizations':
        show_ivi_visualizations_page()

def show_ivi_analysis_page():
    """Display IVI data analysis page (same as ABS)"""
    st.markdown("## 🔍 Job Market Analysis")
    
    # Set proper dataset name for IVI
    dataset_name = st.session_state.get('current_dataset', 'IVI IT Jobs Dataset')
    if dataset_name is None or ('ivi' not in str(dataset_name).lower() and 'anzsco4' not in str(dataset_name).lower()):
        dataset_name = 'IVI IT Jobs Dataset'
    
    st.markdown(f"### 📊 Analysis: {dataset_name}")
    
    # Load and display the dataset
    try:
        # Find the CSV file in data/preprocessed/ (check both direct files and subfolders)
        csv_files = []
        if os.path.exists("data/preprocessed"):
            # Check direct files
            for file in os.listdir("data/preprocessed"):
                if file.endswith(".csv") and "ivi" in file.lower():
                    csv_files.append(os.path.join("data/preprocessed", file))
            
            # Check subfolders for IVI data
            for item in os.listdir("data/preprocessed"):
                item_path = os.path.join("data/preprocessed", item)
                if os.path.isdir(item_path) and "anzsco4" in item.lower():
                    for file in os.listdir(item_path):
                        if file.endswith(".csv") and "4_digit" in file:
                            csv_files.append(os.path.join(item_path, file))
        
        if csv_files:
            # Use the IT jobs CSV file (should be the one with "4_digit" in name)
            it_jobs_file = None
            for csv_file in csv_files:
                if "4_digit" in csv_file:
                    it_jobs_file = csv_file
                    break
            
            if not it_jobs_file:
                it_jobs_file = csv_files[0]  # Fallback to first file
            
            # Load the dataset
            df = pd.read_csv(it_jobs_file)
            
            # Dataset Information (after analysis, before preview)
            st.markdown("### 📊 Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns):,}")
            
            # Data Preview (after dataset info)
            st.markdown("### 📄 Data Preview")
            st.dataframe(df.head(10))
            
        else:
            st.warning("No processed IVI data found. Please click the IVI Dashboard button to download and process the data.")
            
    except Exception as e:
        st.error(f"Error loading IVI data: {e}")

def show_ivi_visualizations_page():
    """Display IVI visualizations page with all 6 charts"""
    st.markdown("## 📈 IVI Visualizations")
    
    # Load IVI data
    try:
        # Find the CSV file in data/preprocessed/
        csv_files = []
        if os.path.exists("data/preprocessed"):
            # Check subfolders for IVI data
            for item in os.listdir("data/preprocessed"):
                item_path = os.path.join("data/preprocessed", item)
                if os.path.isdir(item_path) and "anzsco4" in item.lower():
                    for file in os.listdir(item_path):
                        if file.endswith(".csv") and "4_digit" in file:
                            csv_files.append(os.path.join(item_path, file))
        
        if not csv_files:
            st.warning("No processed IVI data found. Please process the data first.")
            return
            
        # Load the IT jobs data
        it_jobs_file = csv_files[0]
        df = pd.read_csv(it_jobs_file)
        
        # Initialize visualizer
        from agents.data_visualizer import DataVisualizer
        visualizer = DataVisualizer()
        
        # Create visualization tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                    "📈 Trend Over Time", 
                    "🗺️ State Distribution", 
                    "🏆 Top Occupations",
                    "📊 YoY Growth",
                    "🔥 Heatmap",
                    "🦠 COVID Impact",
                    "📈 Growth Rate Summary",
                    "🗺️ Forecast by State",
                    "👥 Forecast by Occupation"
                ])
        
        with tab1:
            st.markdown("### 📈 IT Job Vacancies Trend Over Time")
            st.markdown("Shows how job demand has evolved over time, identifying COVID dips and growth phases.")
            
            chart = visualizer.chart_ivi_trend_over_time(df)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                
                # Show summary stats
                date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                monthly_totals = [df[col].sum() for col in date_columns if col in df.columns]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peak Vacancies", f"{max(monthly_totals):,.0f}")
                with col2:
                    st.metric("Lowest Vacancies", f"{min(monthly_totals):,.0f}")
                with col3:
                    latest = monthly_totals[-1] if monthly_totals else 0
                    previous = monthly_totals[-2] if len(monthly_totals) > 1 else latest
                    change = ((latest - previous) / previous * 100) if previous != 0 else 0
                    st.metric("Latest Change", f"{change:+.1f}%")
            else:
                st.error("Error creating trend chart")
            
        with tab2:
            st.markdown("### 🗺️ IT Job Vacancies by State/Territory")
            st.markdown("Interactive map showing IT job vacancies across Australian states and territories. Hover over regions to see detailed vacancy counts.")
            
            chart = visualizer.chart_ivi_state_distribution(df)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                
                # Show state ranking (exclude AUST - national total)
                numeric_cols = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                state_totals = df[df['state'] != 'AUST'].groupby('state')[numeric_cols].sum().sum(axis=1)
                state_ranking = state_totals.sort_values(ascending=False)
                
                st.markdown("#### 🏆 State Rankings")
                for i, (state, vacancies) in enumerate(state_ranking.items(), 1):
                    st.write(f"{i}. **{state}**: {vacancies:,.0f} vacancies")
            else:
                st.error("Error creating state distribution chart")
            
        with tab3:
            st.markdown("### 🏆 Top IT Occupations by Vacancies")
            st.markdown("Identify the most in-demand IT roles based on total vacancy counts. Displayed as a horizontal bar chart.")
            
            chart = visualizer.chart_ivi_top_occupations(df, top_n=10)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                
                # Show detailed breakdown
                numeric_cols = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                occupation_totals = df.groupby('ANZSCO_TITLE')[numeric_cols].sum().sum(axis=1)
                top_occupations = occupation_totals.nlargest(10)
                
                st.markdown("#### 📊 Detailed Breakdown")
                for i, (occupation, vacancies) in enumerate(top_occupations.items(), 1):
                    percentage = (vacancies / occupation_totals.sum()) * 100
                    st.write(f"{i}. **{occupation}**: {vacancies:,.0f} vacancies ({percentage:.1f}%)")
            else:
                st.error("Error creating top occupations chart")
            
        with tab4:
            st.markdown("### 📊 Year-on-Year Growth in IT Job Vacancies")
            st.markdown("Highlights boom or decline periods in IT job market.")
            
            chart = visualizer.chart_ivi_yoy_growth(df)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                
                # Show growth summary
                date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                dates = pd.to_datetime(date_columns, errors='coerce')
                years = pd.Series([d.year if pd.notna(d) else None for d in dates])
                
                annual_totals = {}
                for i, col in enumerate(date_columns):
                    if pd.notna(years[i]):
                        year = int(years[i])
                        if year not in annual_totals:
                            annual_totals[year] = 0
                        annual_totals[year] += df[col].sum()
                
                years_sorted = sorted(annual_totals.keys())
                yoy_growth = []
                for i in range(1, len(years_sorted)):
                    current_total = annual_totals[years_sorted[i]]
                    previous_total = annual_totals[years_sorted[i-1]]
                    if previous_total > 0:
                        growth = ((current_total - previous_total) / previous_total) * 100
                        yoy_growth.append(growth)
                
                if yoy_growth:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Highest Growth", f"{max(yoy_growth):+.1f}%")
                    with col2:
                        st.metric("Biggest Decline", f"{min(yoy_growth):+.1f}%")
                    with col3:
                        avg_growth = sum(yoy_growth) / len(yoy_growth)
                        st.metric("Average Growth", f"{avg_growth:+.1f}%")
            else:
                st.error("Error creating YoY growth chart")
            
        with tab5:
            st.markdown("### 🔥 IT Vacancies: Occupation vs State Heatmap")
            st.markdown("Detect regional specialization patterns across IT occupations and states.")
            
            chart = visualizer.chart_ivi_occupation_state_heatmap(df)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                
                # Show insights
                numeric_cols = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                heatmap_data = df.groupby(['ANZSCO_TITLE', 'state'])[numeric_cols].sum().sum(axis=1)
                heatmap_pivot = heatmap_data.unstack(fill_value=0)
                
                st.markdown("#### 🔍 Key Insights")
                max_vacancies = heatmap_pivot.max().max()
                max_location = heatmap_pivot.stack().idxmax()
                st.write(f"**Highest vacancy concentration**: {max_location[0]} in {max_location[1]} ({max_vacancies:,.0f} vacancies)")
                
                # Show top 3 combinations
                top_combinations = heatmap_pivot.stack().nlargest(3)
                st.write("**Top 3 Occupation-State combinations:**")
                for i, ((occupation, state), vacancies) in enumerate(top_combinations.items(), 1):
                    st.write(f"{i}. {occupation} in {state}: {vacancies:,.0f} vacancies")
            else:
                st.error("Error creating heatmap")
            
        with tab6:
            st.markdown("### 🦠 COVID Impact on IT Job Market")
            st.markdown("Highlight the fall and recovery phases for IT roles during COVID-19.")
            
            chart = visualizer.chart_ivi_covid_impact(df)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                
                # Calculate impact metrics
                date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                dates = pd.to_datetime(date_columns, errors='coerce')
                
                before_covid = []
                during_covid = []
                after_covid = []
                
                for i, col in enumerate(date_columns):
                    if i < len(dates) and pd.notna(dates[i]):
                        date = dates[i]
                        covid_start = pd.Timestamp('2020-03-01')
                        covid_end = pd.Timestamp('2022-01-01')
                        if date < covid_start:
                            before_covid.append(df[col].sum())
                        elif date >= covid_start and date < covid_end:
                            during_covid.append(df[col].sum())
                        else:
                            after_covid.append(df[col].sum())
                
                before_avg = sum(before_covid) / len(before_covid) if before_covid else 0
                during_avg = sum(during_covid) / len(during_covid) if during_covid else 0
                after_avg = sum(after_covid) / len(after_covid) if after_covid else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    decline = ((during_avg - before_avg) / before_avg * 100) if before_avg > 0 else 0
                    st.metric("COVID Impact", f"{decline:+.1f}%", delta=f"{decline:+.1f}%")
                with col2:
                    recovery = ((after_avg - during_avg) / during_avg * 100) if during_avg > 0 else 0
                    st.metric("Recovery Rate", f"{recovery:+.1f}%", delta=f"{recovery:+.1f}%")
                with col3:
                    net_change = ((after_avg - before_avg) / before_avg * 100) if before_avg > 0 else 0
                    st.metric("Net Change", f"{net_change:+.1f}%", delta=f"{net_change:+.1f}%")
                
                # Show timeline
                st.markdown("#### 📅 Timeline Analysis")
                st.write(f"**Pre-COVID Average**: {before_avg:,.0f} monthly vacancies")
                st.write(f"**During COVID Average**: {during_avg:,.0f} monthly vacancies")
                st.write(f"**Post-COVID Average**: {after_avg:,.0f} monthly vacancies")
            else:
                st.error("Error creating COVID impact chart")
        
        with tab7:
            st.markdown("### 📈 IT Job Vacancies: Growth Rate Summary")
            st.markdown("Visualize percentage growth between last historical point and forecasted endpoint.")
            
            # Forecast periods selector
            forecast_periods = st.slider("Forecast Periods (months)", 3, 24, 8, key="forecast_periods_growth")
            
            # Growth rate summary
            try:
                date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                monthly_totals = [df[col].sum() for col in date_columns if col in df.columns]
                
                if len(monthly_totals) >= 12:
                    import numpy as np
                    import plotly.graph_objects as go
                    from sklearn.linear_model import LinearRegression

                    # Use simple numeric x-axis
                    X = np.arange(len(monthly_totals)).reshape(-1, 1)
                    y = np.array(monthly_totals)
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Generate forecast
                    forecast_X = np.arange(len(monthly_totals), len(monthly_totals) + forecast_periods).reshape(-1, 1)
                    forecast_y = model.predict(forecast_X)
                    
                    # Create simple x-axis labels
                    historical_x = list(range(len(monthly_totals)))
                    forecast_x = list(range(len(monthly_totals), len(monthly_totals) + forecast_periods))
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=historical_x,
                        y=monthly_totals,
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=6, color='#667eea')
                    ))
                    
                    # Forecast data
                    fig.add_trace(go.Scatter(
                        x=forecast_x,
                        y=forecast_y,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#ff6b6b', width=3, dash='dash'),
                        marker=dict(size=6, color='#ff6b6b')
                    ))
                    
                    # Add growth rate annotation
                    last_historical = monthly_totals[-1]
                    forecast_endpoint = forecast_y[-1]
                    growth_rate = ((forecast_endpoint - last_historical) / last_historical) * 100
                    
                    fig.add_annotation(
                        x=forecast_x[-1],
                        y=forecast_endpoint,
                        text=f"Growth: {growth_rate:+.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="green" if growth_rate > 0 else "red",
                        ax=0, ay=-40,
                        bgcolor="white",
                        bordercolor="gray",
                        font=dict(size=12, color='#2d3748')
                    )
                    
                    fig.update_layout(
                        title="IT Job Vacancies: Growth Rate Summary",
                        xaxis_title="Time Period",
                        yaxis_title="Job Vacancies",
                        hovermode='x unified',
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#2d3748'),
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Show growth rate metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Historical End", f"{last_historical:,.0f}")
                    with col2:
                        st.metric("Forecast End", f"{forecast_endpoint:,.0f}")
                    with col3:
                        st.metric("Growth Rate", f"{growth_rate:+.1f}%", 
                                delta=f"{growth_rate:+.1f}%" if growth_rate != 0 else None)
                else:
                    st.error("Insufficient data for forecasting")
            except Exception as e:
                st.error(f"Error creating growth rate summary: {e}")
        
        with tab8:
            st.markdown("### 🗺️ IT Job Vacancies: Forecast by State")
            st.markdown("Forecast future vacancies separately for each state with individual trend lines.")
            
            # Forecast periods selector
            forecast_periods = st.slider("Forecast Periods (months)", 3, 24, 8, key="forecast_periods_state")
            
            # Forecast by state
            try:
                import numpy as np
                import plotly.graph_objects as go
                from sklearn.linear_model import LinearRegression
                
                date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                
                fig = go.Figure()
                
                # Get unique states (excluding AUST)
                states = [state for state in df['state'].unique() if state != 'AUST']
                colors = ['#667eea', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
                
                for i, state in enumerate(states[:8]):  # Limit to 8 states for readability
                    state_data = df[df['state'] == state]
                    state_monthly = []
                    
                    for col in date_columns:
                        state_monthly.append(state_data[col].sum())
                    
                    if len(state_monthly) >= 12:
                        # Historical data
                        historical_x = list(range(len(state_monthly)))
                        fig.add_trace(go.Scatter(
                            x=historical_x,
                            y=state_monthly,
                            mode='lines+markers',
                            name=f'{state} (Historical)',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Forecast
                        X = np.arange(len(state_monthly)).reshape(-1, 1)
                        y = np.array(state_monthly)
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        forecast_X = np.arange(len(state_monthly), len(state_monthly) + forecast_periods).reshape(-1, 1)
                        forecast_y = model.predict(forecast_X)
                        
                        # Forecast x-axis
                        forecast_x = list(range(len(state_monthly), len(state_monthly) + forecast_periods))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_x,
                            y=forecast_y,
                            mode='lines+markers',
                            name=f'{state} (Forecast)',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                            marker=dict(size=4)
                        ))
                
                fig.update_layout(
                    title="IT Job Vacancies: Forecast by State",
                    xaxis_title="Time Period",
                    yaxis_title="Job Vacancies",
                    hovermode='x unified',
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748'),
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Show state forecast summary
                st.markdown("#### 📊 State Forecast Summary")
                state_forecasts = {}
                
                for state in states:
                    state_data = df[df['state'] == state]
                    state_monthly = []
                    
                    for col in date_columns:
                        state_monthly.append(state_data[col].sum())
                    
                    if len(state_monthly) >= 12:
                        X = np.arange(len(state_monthly)).reshape(-1, 1)
                        y = np.array(state_monthly)
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        last_value = state_monthly[-1]
                        forecast_X = np.array([[len(state_monthly) + forecast_periods - 1]])
                        forecast_value = model.predict(forecast_X)[0]
                        growth = ((forecast_value - last_value) / last_value * 100) if last_value > 0 else 0
                        
                        state_forecasts[state] = {
                            'current': last_value,
                            'forecast': forecast_value,
                            'growth': growth
                        }
                
                # Display top 5 states by growth
                if state_forecasts:
                    sorted_states = sorted(state_forecasts.items(), key=lambda x: x[1]['growth'], reverse=True)
                    
                    for i, (state, data) in enumerate(sorted_states[:5], 1):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**{i}. {state}**")
                        with col2:
                            st.write(f"Current: {data['current']:,.0f}")
                        with col3:
                            st.write(f"Forecast: {data['forecast']:,.0f}")
                        with col4:
                            st.write(f"Growth: {data['growth']:+.1f}%")
            except Exception as e:
                st.error(f"Error creating forecast by state chart: {e}")
        
        with tab9:
            st.markdown("### 👥 IT Job Vacancies: Forecast by Occupation")
            st.markdown("Forecast specific IT occupations with individual trend analysis.")
            
            # Forecast parameters
            col1, col2 = st.columns(2)
            with col1:
                forecast_periods = st.slider("Forecast Periods (months)", 3, 24, 8, key="forecast_periods_occupation")
            with col2:
                top_n = st.slider("Top N Occupations", 3, 10, 5, key="top_n_occupations")
            
            # Forecast by occupation
            try:
                import numpy as np
                import plotly.graph_objects as go
                from sklearn.linear_model import LinearRegression
                
                date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
                
                # Get top occupations by total vacancies
                occupation_totals = df.groupby('ANZSCO_TITLE')[date_columns].sum().sum(axis=1)
                top_occupations = occupation_totals.nlargest(top_n).index.tolist()
                
                fig = go.Figure()
                colors = ['#667eea', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
                
                for i, occupation in enumerate(top_occupations):
                    occ_data = df[df['ANZSCO_TITLE'] == occupation]
                    occ_monthly = []
                    
                    for col in date_columns:
                        occ_monthly.append(occ_data[col].sum())
                    
                    if len(occ_monthly) >= 12:
                        # Historical data
                        historical_x = list(range(len(occ_monthly)))
                        fig.add_trace(go.Scatter(
                            x=historical_x,
                            y=occ_monthly,
                            mode='lines+markers',
                            name=f'{occupation[:30]}... (Historical)' if len(occupation) > 30 else f'{occupation} (Historical)',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Forecast
                        X = np.arange(len(occ_monthly)).reshape(-1, 1)
                        y = np.array(occ_monthly)
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        forecast_X = np.arange(len(occ_monthly), len(occ_monthly) + forecast_periods).reshape(-1, 1)
                        forecast_y = model.predict(forecast_X)
                        
                        # Forecast x-axis
                        forecast_x = list(range(len(occ_monthly), len(occ_monthly) + forecast_periods))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_x,
                            y=forecast_y,
                            mode='lines+markers',
                            name=f'{occupation[:30]}... (Forecast)' if len(occupation) > 30 else f'{occupation} (Forecast)',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                            marker=dict(size=4)
                        ))
                
                fig.update_layout(
                    title="IT Job Vacancies: Forecast by Occupation",
                    xaxis_title="Time Period",
                    yaxis_title="Job Vacancies",
                    hovermode='x unified',
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748'),
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Show occupation forecast summary
                st.markdown("#### 📊 Top Occupation Forecast Summary")
                occupation_forecasts = {}
                
                for occupation in top_occupations:
                    occ_data = df[df['ANZSCO_TITLE'] == occupation]
                    occ_monthly = []
                    
                    for col in date_columns:
                        occ_monthly.append(occ_data[col].sum())
                    
                    if len(occ_monthly) >= 12:
                        X = np.arange(len(occ_monthly)).reshape(-1, 1)
                        y = np.array(occ_monthly)
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        last_value = occ_monthly[-1]
                        forecast_X = np.array([[len(occ_monthly) + forecast_periods - 1]])
                        forecast_value = model.predict(forecast_X)[0]
                        growth = ((forecast_value - last_value) / last_value * 100) if last_value > 0 else 0
                        
                        occupation_forecasts[occupation] = {
                            'current': last_value,
                            'forecast': forecast_value,
                            'growth': growth
                        }
                
                # Display occupation forecasts
                if occupation_forecasts:
                    sorted_occupations = sorted(occupation_forecasts.items(), key=lambda x: x[1]['growth'], reverse=True)
                    
                    for i, (occupation, data) in enumerate(sorted_occupations, 1):
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            st.write(f"**{i}. {occupation[:50]}{'...' if len(occupation) > 50 else ''}**")
                        with col2:
                            st.write(f"Current: {data['current']:,.0f}")
                        with col3:
                            st.write(f"Forecast: {data['forecast']:,.0f}")
                        with col4:
                            st.write(f"Growth: {data['growth']:+.1f}%")
            except Exception as e:
                st.error(f"Error creating forecast by occupation chart: {e}")
                         
    except Exception as e:
        st.error(f"Error loading visualizations: {e}")


def show_analysis_page():
    """Show the main data analysis page"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Job Market Analysis</h1>', unsafe_allow_html=True)

    # Bootstrap datasets from backend on first load or after refresh
    try:
        ds = requests.get("http://localhost:8000/datasets", timeout=10)
        if ds.status_code == 200:
            names = ds.json().get("datasets", [])
            if not names:
                # ask backend to restore from disk if empty
                r = requests.post("http://localhost:8000/restore-datasets", timeout=20)
                if r.status_code == 200:
                    names = r.json().get("datasets", [])
            # ensure session state reflects backend
            st.session_state.datasets = {name: {"name": name} for name in names}
            if names and not st.session_state.get('current_dataset'):
                st.session_state.current_dataset = names[0]
    except Exception:
        pass
    

    
    # Main content area
    if not st.session_state.datasets or len(st.session_state.datasets) == 0:
        show_welcome_screen()
    else:
        show_dataset_analysis()
    
    # Footer
    st.divider()
    st.markdown(
        "---\n"
        "**Job Market Analysis Agentic AI**"
    )

def show_visualizations_page():
    """Show the visualizations page"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Visualizations</h1>', unsafe_allow_html=True)
    

    # Discover preprocessed industry CSVs (Data1.csv inside dataset _sheets folder)
    pre_csvs = []
    try:
        for root, dirs, files in os.walk(os.path.join("data", "preprocessed")):
            for f in files:
                if f.lower() == "data1.csv":
                    pre_csvs.append(os.path.join(root, f))
    except Exception:
        pass

    viz = DataVisualizer()

    if not pre_csvs:
        st.info("No preprocessed industry CSVs found yet. Download/process an ABS dataset first.")
    else:
        # Filter to industry-only CSVs and create a tab per dataset
        try:
            pre_csvs = sorted(pre_csvs, key=lambda p: os.path.getmtime(p), reverse=True)
        except Exception:
            pre_csvs = sorted(pre_csvs)

        known_inds = {"Mining","Manufacturing","Construction","Retail Trade","Accommodation and Food Services",
                      "Administrative and Support Services","Education and Training",
                      "Electricity, Gas, Water and Waste Services","Health Care and Social Assistance"}
        industry_csvs = []
        for p in pre_csvs:
            try:
                head = pd.read_csv(p, nrows=1)
                cols = set(map(str, head.columns))
                if ("Date" in cols) and (len(cols & known_inds) >= 3):
                    industry_csvs.append(p)
            except Exception:
                continue
        if not industry_csvs:
            st.info("No industry-formatted datasets detected yet.")
            return

        labels = [os.path.basename(os.path.dirname(p)) for p in industry_csvs]
        tabs = st.tabs(labels)

        for idx, (tab, csv_path) in enumerate(zip(tabs, industry_csvs)):
            with tab:

                try:
                    df = pd.read_csv(csv_path)
                    long = viz.prepare_long_format(df)
                    industries = sorted(long["Industry"].unique())
                except Exception as e:
                    long = pd.DataFrame(columns=["Date","Industry","Value","Year"])  # empty
                    industries = []
                    st.error(f"Failed to load dataset: {e}")

                chosen = st.multiselect("Industries", industries, default=industries[:6], key=f"inds-{idx}")


                st.subheader("Rankings & Composition")
                r1, r2 = st.columns(2)
                with r1:
                    fig_bar = viz.chart_latest_bar(long)
                    st.plotly_chart(fig_bar, config={'displayModeBar': False}, key=f"pl-bar-{idx}")
                with r2:
                    fig_pie = viz.chart_latest_pie(long)
                    st.plotly_chart(fig_pie, config={'displayModeBar': False}, key=f"pl-pie-{idx}")
                fig_stack = viz.chart_stacked_composition(long)
                st.plotly_chart(fig_stack, config={'displayModeBar': False}, key=f"pl-stack-{idx}")

                st.subheader("Growth & Change")
                g1, g2 = st.columns(2)
                with g1:
                    fig_yoy = viz.chart_yoy_heatmap(long)
                    st.plotly_chart(fig_yoy, config={'displayModeBar': False}, key=f"pl-yoy-{idx}")
                with g2:
                    fig_bubble = viz.chart_growth_vs_size_bubble(long)
                    st.plotly_chart(fig_bubble, config={'displayModeBar': False}, key=f"pl-bubble-{idx}")

                st.subheader("Future Outlook (5-Year Forecast)")
                
                # Forecast controls
                col_forecast_years, col_uncertainty = st.columns([1, 1])
                with col_forecast_years:
                    forecast_years = st.number_input("Forecast Years", min_value=1, max_value=10, value=5, step=1, key=f"forecast-years-{idx}")
                with col_uncertainty:
                    prediction_style = st.selectbox(
                        "Prediction Style", 
                        ["Optimistic", "Balanced", "Conservative"], 
                        index=1, 
                        help="Optimistic = Narrower range (more precise), Conservative = Wider range (more cautious)",
                        key=f"prediction-style-{idx}"
                    )
                    # Convert to confidence level
                    confidence_level = {"Optimistic": 0.90, "Balanced": 0.95, "Conservative": 0.99}[prediction_style]
                
                # Main forecast chart
                fig_forecast = viz.chart_historical_and_forecast(long, industries=chosen, forecast_years=forecast_years, confidence_level=confidence_level)
                st.plotly_chart(fig_forecast, config={'displayModeBar': False}, key=f"pl-forecast-{idx}")
                
                # Forecast summary (removed R² score chart for better UX)
                # fig_forecast_summary = viz.chart_forecast_summary(long, industries=chosen, forecast_years=forecast_years)
                # st.plotly_chart(fig_forecast_summary, config={'displayModeBar': False}, key=f"pl-forecast-summary-{idx}")
                
                # Forecast disclaimer
                st.info("⚠️ **Forecast Disclaimer**: Predictions are based on historical trends and should be used for planning purposes only. Actual results may vary due to unforeseen economic conditions, policy changes, or market disruptions.")

                st.subheader("COVID Impact")
                fig_covid = viz.chart_indexed(long, base="2019-01-01", industries=chosen)
                st.plotly_chart(fig_covid, config={'displayModeBar': False}, key=f"pl-indexed-covid-{idx}")
                # end dataset tab content

    st.divider()
    
    # Footer
    st.divider()
    st.markdown(
        "---\n"
        "**Job Market Analysis Agentic AI** "
    )

def show_welcome_screen():
    """Display welcome screen when no datasets are loaded"""
    
    # Use Streamlit's native components instead of HTML
    st.markdown("## 🔍 Job Market Analysis")
    st.markdown("### Welcome to Job Market Analysis AI!")
    
    st.markdown("---")
    
    st.markdown("#### This intelligent AI agent can automatically:")
    st.markdown("""
    - 🔍 **Discover** datasets from URLs
    - 📊 **Preprocess** and clean data  
    - 🧠 **Analyze** job market trends
    """)
    
    st.markdown("#### Getting Started:")
    st.markdown("""
    1. **Enter a URL** in the sidebar to analyze existing datasets
    2. **Upload a file** to analyze your own data
    3. **Explore insights** and visualizations
    """)
    
    st.markdown("#### Example URLs:")
    st.markdown("""
    - Australian Bureau of Statistics (ABS) job vacancies
    - Government employment data
    - Industry reports and surveys
    """)
    
    st.markdown("---")
    
    # Call to action box using st.info
    st.info("🚀 **Try it out!** Click on the ABS Dataset Dashboard button above to automatically download and analyze the latest job market data!")


def show_dataset_analysis():
    """Display analysis results for the selected dataset"""
    
    if not st.session_state.current_dataset:
        return
    
    dataset_name = st.session_state.current_dataset
    analysis = st.session_state.analysis_results.get(dataset_name, {})
    
    # Header
    st.header(f"📊 Analysis: {dataset_name}")
    
    # Overview metrics
    if analysis and 'overview' in analysis:
        overview = analysis['overview']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records",
                overview.get('total_records', 'N/A'),
                help="Total number of records in the dataset"
            )
        
        with col2:
            st.metric(
                "Columns",
                overview.get('data_coverage', {}).get('total_columns', 'N/A'),
                help="Total number of columns"
            )
        
        with col3:
            completeness = overview.get('data_coverage', {}).get('completeness_rate', 0)
            st.metric(
                "Data Quality",
                f"{completeness:.1f}%",
                help="Percentage of complete records"
            )
        
        with col4:
            if 'key_metrics' in overview:
                metrics = overview['key_metrics']
                # Find first numeric metric
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and 'total' in key:
                        st.metric(
                            "Total Vacancies",
                            f"{value:,.0f}",
                            help="Total job vacancies"
                        )
                        break
                else:
                    st.metric("Total Vacancies", "N/A")
    
    # Key Insights
    if analysis and 'key_insights' in analysis:
        st.subheader("💡 Key Insights")
        insights = analysis['key_insights']
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    # Recommendations
    if analysis and 'recommendations' in analysis:
        st.subheader("🎯 Recommendations")
        recommendations = analysis['recommendations']
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # Detailed Analysis Tabs
    if analysis:
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🗺️ Geography", "🏭 Industry", "🏛️ Sector"])
        
        with tab1:
            show_trends_analysis(analysis)
        
        with tab2:
            show_geography_analysis(analysis)
        
        with tab3:
            show_industry_analysis(analysis)
        
        with tab4:
            show_sector_analysis(analysis)
    
    # Data Preview
    st.subheader("📋 Data Preview")
    # Check if dataset has multiple sheets
    try:
        # Try different name variations to find the correct one
        possible_names = [
            dataset_name,  # Original name
            dataset_name.replace(' ', '_'),  # Replace spaces with underscores
            dataset_name.replace('[', '[').replace(']', ']'),  # Keep brackets
            dataset_name.replace(' ', ''),  # Remove all spaces
        ]
        
        sheets_info = None
        working_name = None
        
        for name_variant in possible_names:
            try:
                # Properly encode the dataset name for the URL
                import urllib.parse
                encoded_name = urllib.parse.quote(name_variant, safe='')
                
                sheets_response = requests.get(f"http://localhost:8000/sheets/{encoded_name}")
                
                if sheets_response.status_code == 200:
                    sheets_info = sheets_response.json()
                    working_name = name_variant
                    break
                    
            except Exception:
                continue
        
        if sheets_info:
            if sheets_info.get('has_multiple_sheets', False):
                # Create tabs for multiple sheets
                sheet_names = sheets_info.get('sheet_names', [])
                
                if len(sheet_names) > 1:
                    tabs = st.tabs(sheet_names)
                    
                    for i, (tab, sheet_name) in enumerate(zip(tabs, sheet_names)):
                        with tab:
                            st.write(f"**Sheet: {sheet_name}**")
                            sheet_data = sheets_info.get('sheets', {}).get(sheet_name, {})
                            
                            if 'sample_data' in sheet_data and sheet_data['sample_data']:
                                df_sample = pd.DataFrame(sheet_data['sample_data'])
                                st.write(f"**Shape:** {sheet_data.get('shape', 'Unknown')}")
                                st.write(f"**Columns:** {len(sheet_data.get('columns', []))}")
                                st.dataframe(df_sample, width='stretch')
                            else:
                                st.info(f"No sample data available for sheet '{sheet_name}'")
                else:
                    # Single sheet
                    show_single_sheet_preview(working_name or dataset_name)
            else:
                # Single sheet dataset
                show_single_sheet_preview(working_name or dataset_name)
        else:
            st.error("❌ Could not find dataset in backend")
    except Exception as e:
        st.error(f"❌ Error loading sheet data: {str(e)}")
        # Fallback to single sheet preview
        show_single_sheet_preview(dataset_name)

def show_trends_analysis(analysis):
    """Display trends analysis"""
    
    trends = analysis.get('trends', {})
    temporal_trends = trends.get('temporal_trends', {})
    
    if temporal_trends:
        st.write("### Temporal Trends")
        
        for metric, trend_info in temporal_trends.items():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Metric", metric)
            
            with col2:
                direction = trend_info.get('trend_direction', 'unknown')
                st.metric("Trend", direction.title())
            
            with col3:
                strength = trend_info.get('trend_strength', 0)
                st.metric("Strength", f"{strength:.2f}")
    else:
        st.info("No trend data available")

def show_geography_analysis(analysis):
    """Display geographic analysis"""
    
    geography = analysis.get('geographic_analysis', {})
    states_territories = geography.get('states_territories', {})
    
    if states_territories:
        st.write("### Geographic Distribution")
        
        for col, data in states_territories.items():
            if 'top_locations' in data:
                top_locations = data['top_locations']
                
                # Display as table
                df_locations = pd.DataFrame(list(top_locations.items()), columns=['Location', 'Count'])
                st.dataframe(df_locations, width='stretch')
    else:
        st.info("No geographic data available")

def show_industry_analysis(analysis):
    """Display industry analysis"""
    
    industry = analysis.get('industry_analysis', {})
    top_industries = industry.get('top_industries', {})
    
    if top_industries:
        st.write("### Industry Distribution")
        
        for col, data in top_industries.items():
            if 'names' in data and 'values' in data:
                names = data['names']
                values = data['values']
                
                # Display as table
                df_industries = pd.DataFrame({'Industry': names, 'Count': values})
                st.dataframe(df_industries, width='stretch')
    else:
        st.info("No industry data available")

def show_sector_analysis(analysis):
    """Display sector analysis"""
    
    sector = analysis.get('sector_analysis', {})
    sector_breakdown = sector.get('sector_breakdown', {})
    
    if sector_breakdown:
        st.write("### Sector Distribution")
        
        for col, data in sector_breakdown.items():
            if 'sectors' in data:
                sectors = data['sectors']
                
                # Display as table
                df_sectors = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Count'])
                st.dataframe(df_sectors, width='stretch')
    else:
        st.info("No sector data available")

def show_single_sheet_preview(dataset_name):
    """Display single sheet preview"""
    try:
        # Properly encode the dataset name for the URL
        import urllib.parse
        encoded_dataset_name = urllib.parse.quote(dataset_name, safe='')
        
        response = requests.get(f"http://localhost:8000/dataset/{encoded_dataset_name}")
        if response.status_code == 200:
            data_info = response.json()
            if 'sample_data' in data_info and data_info['sample_data']:
                df_sample = pd.DataFrame(data_info['sample_data'])
                st.dataframe(df_sample, width='stretch')
            else:
                st.info("Sample data not available")
        else:
            st.info("Sample data not available")
    except:
        st.info("Sample data not available")


if __name__ == "__main__":
    main()
