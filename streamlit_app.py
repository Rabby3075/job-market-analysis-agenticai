"""
Streamlit Frontend for Job Market Analysis
Provides an interactive interface for the agentic AI system
"""

import io
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Job Market Analysis AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Job Market Analysis Agentic AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Quick Actions")
        
        # URL Analysis
        st.subheader("Analyze URL")
        url_input = st.text_input(
            "Enter dataset URL:",
            placeholder="https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia/may-2025",
            help="Enter a URL containing job market datasets"
        )
        
        if st.button("🔍 Discover & Analyze", type="primary"):
            if url_input:
                analyze_url(url_input)
            else:
                st.error("Please enter a URL")
        
        st.divider()
        
        # File Upload
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload your own dataset file"
        )
        
        if uploaded_file is not None:
            if st.button("📊 Process Uploaded File"):
                process_uploaded_file(uploaded_file)
        
        st.divider()
        
        # Dataset Selection
        if st.session_state.datasets:
            st.subheader("📁 Available Datasets")
            
            # Handle both list and dictionary formats
            if isinstance(st.session_state.datasets, list):
                dataset_names = st.session_state.datasets
            else:
                dataset_names = list(st.session_state.datasets.keys())
            
            if dataset_names:
                # Safe index selection
                try:
                    if st.session_state.current_dataset and st.session_state.current_dataset in dataset_names:
                        default_index = dataset_names.index(st.session_state.current_dataset)
                    else:
                        default_index = 0
                except (ValueError, AttributeError):
                    default_index = 0
                    st.session_state.current_dataset = None
                
                selected_dataset = st.selectbox(
                    "Select dataset to view:",
                    dataset_names,
                    index=default_index
                )
            
            if selected_dataset != st.session_state.current_dataset:
                st.session_state.current_dataset = selected_dataset
                st.rerun()
    
    # Main content area
    if not st.session_state.datasets or len(st.session_state.datasets) == 0:
        show_welcome_screen()
    else:
        show_dataset_analysis()
    
    # Footer
    st.divider()
    st.markdown(
        "---\n"
        "**Job Market Analysis Agentic AI** | Built with Streamlit, FastAPI, and AI Agents"
    )

def show_welcome_screen():
    """Display welcome screen when no datasets are loaded"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 Welcome to Job Market Analysis AI!
        
        This intelligent AI agent can automatically:
        
        - **🔍 Discover** datasets from URLs
        - **📊 Preprocess** and clean data
        - **🧠 Analyze** job market trends
        - **📈 Visualize** insights interactively
        
        ### Getting Started:
        
        1. **Enter a URL** in the sidebar to analyze existing datasets
        2. **Upload a file** to analyze your own data
        3. **Explore insights** and visualizations
        
        ### Example URLs:
        - Australian Bureau of Statistics (ABS) job vacancies
        - Government employment data
        - Industry reports and surveys
        """)
        
        # Example analysis
        st.markdown("### 🚀 Try it out!")
        st.markdown("Use the sidebar to enter a URL or upload a file to get started with your analysis.")

def analyze_url(url):
    """Analyze datasets from a given URL"""
    
    with st.spinner("🔍 Discovering datasets..."):
        try:
            # Call the FastAPI backend
            response = requests.post(
                "http://localhost:8000/analyze-url",
                json={"url": url},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    st.success(f"✅ Successfully discovered {data['datasets_discovered']} datasets!")
                    
                    # Store results in session state
                    st.session_state.datasets = data.get("processed_datasets", [])
                    st.session_state.analysis_results = data.get("sample_analysis", {})
                    
                    # Convert datasets to a dictionary format for easier handling
                    if isinstance(st.session_state.datasets, list):
                        st.session_state.datasets = {name: {"name": name} for name in st.session_state.datasets}
                    
                    # Show results
                    st.json(data)
                    st.rerun()
                else:
                    st.error("❌ Analysis failed")
                    st.json(data)
            else:
                st.error(f"❌ Error: {response.status_code}")
                st.text(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Please ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def process_uploaded_file(uploaded_file):
    """Process an uploaded dataset file"""
    
    with st.spinner("📊 Processing uploaded file..."):
        try:
            # Save file temporarily
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Call the FastAPI backend
            files = {"file": (uploaded_file.name, open(file_path, "rb"), uploaded_file.type)}
            response = requests.post(
                "http://localhost:8000/upload-dataset",
                files=files,
                timeout=30
            )
            
            # Clean up temp file
            os.remove(file_path)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    st.success(f"✅ Successfully processed {uploaded_file.name}!")
                    
                    # Store results in session state
                    dataset_name = data['dataset_name']
                    
                    # Ensure datasets is a dictionary
                    if not isinstance(st.session_state.datasets, dict):
                        st.session_state.datasets = {}
                    
                    st.session_state.datasets[dataset_name] = data
                    st.session_state.analysis_results[dataset_name] = data['analysis']
                    st.session_state.current_dataset = dataset_name
                    
                    st.json(data)
                    st.rerun()
                else:
                    st.error("❌ Processing failed")
                    st.json(data)
            else:
                st.error(f"❌ Error: {response.status_code}")
                st.text(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Please ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

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
                    
            except Exception as e:
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
                                st.dataframe(df_sample, use_container_width=True)
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
                
                # Create bar chart
                fig = px.bar(
                    x=list(top_locations.keys()),
                    y=list(top_locations.values()),
                    title=f"Top Locations - {col}",
                    labels={'x': 'Location', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
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
                
                # Create horizontal bar chart
                fig = px.bar(
                    y=names,
                    x=values,
                    orientation='h',
                    title=f"Top Industries - {col}",
                    labels={'x': 'Count', 'y': 'Industry'}
                )
                st.plotly_chart(fig, use_container_width=True)
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
                
                # Create pie chart
                fig = px.pie(
                    values=list(sectors.values()),
                    names=list(sectors.keys()),
                    title=f"Sector Breakdown - {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
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
                st.dataframe(df_sample, use_container_width=True)
            else:
                st.info("Sample data not available")
        else:
            st.info("Sample data not available")
    except:
        st.info("Sample data not available")

if __name__ == "__main__":
    main()
