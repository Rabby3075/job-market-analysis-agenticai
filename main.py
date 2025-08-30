"""
Job Market Analysis Agentic AI - FastAPI Backend
Main application orchestrating the AI agents
"""

import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agents.analyzer import JobMarketAnalyzer
# Import our AI agents
from agents.data_discovery import DataDiscoveryAgent
from agents.data_processor import DataProcessorAgent
from utils.json_helper import clean_analysis_results, convert_to_serializable
from utils.logger import get_logger
from utils.visualizer import JobMarketVisualizer

# Initialize professional logger
logger = get_logger("main")

# Initialize FastAPI app
app = FastAPI(
    title="Job Market Analysis Agentic AI",
    description="AI-powered job market analysis using specialized agents",
    version="1.0.0"
)

# Initialize AI agents
discovery_agent = DataDiscoveryAgent()
processor_agent = DataProcessorAgent()
analyzer_agent = JobMarketAnalyzer()
visualizer = JobMarketVisualizer()

# Global storage for processed data and results
processed_datasets = {}
analysis_results = {}
visualization_charts = {}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """API information page"""
    return """
    <html>
        <head>
            <title>Job Market Analysis Agentic AI - API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                .api-section { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }
                code { background: #e9ecef; padding: 2px 6px; border-radius: 4px; }
                .streamlit-link { text-align: center; margin: 30px 0; }
                .streamlit-link a { background: #ff4b4b; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; }
                .streamlit-link a:hover { background: #e63939; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Job Market Analysis Agentic AI</h1>
                <p style="text-align: center; color: #666; font-size: 18px;">
                    An intelligent AI system for analyzing job market datasets using specialized agents
                </p>
                
                <div class="streamlit-link">
                    <a href="http://localhost:8501" target="_blank">🚀 Launch Streamlit Interface</a>
                </div>
                
                <div class="api-section">
                    <h3>📚 API Endpoints</h3>
                    <p>Use these endpoints to interact with the AI system programmatically:</p>
                    <ul>
                        <li><code>POST /analyze-url</code> - Analyze datasets from a URL</li>
                        <li><code>POST /upload</code> - Upload and analyze a file</li>
                        <li><code>GET /datasets</code> - List processed datasets</li>
                        <li><code>GET /dataset/{name}</code> - Get dataset details</li>
                        <li><code>GET /sheets/{name}</code> - Get multi-sheet information</li>
                        <li><code>GET /analysis/{name}</code> - Get analysis results</li>
                        <li><code>GET /health</code> - System health check</li>
                    </ul>
                </div>
                
                <div class="api-section">
                    <h3>🔗 Quick Links</h3>
                    <ul>
                        <li><a href="/docs" target="_blank">📖 Interactive API Documentation</a></li>
                        <li><a href="/redoc" target="_blank">📋 Alternative API Documentation</a></li>
                        <li><a href="http://localhost:8501" target="_blank">🎨 Streamlit User Interface</a></li>
                    </ul>
                </div>
                
                <div class="api-section">
                    <h3>🎯 For ABS Job Vacancies Data</h3>
                    <p>Use the Streamlit interface to analyze:</p>
                    <ul>
                        <li>Job vacancies by states and territories</li>
                        <li>Private sector job vacancies</li>
                        <li>Public sector job vacancies</li>
                        <li>Job vacancies by industry</li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents": {
            "discovery": "active",
            "processor": "active", 
            "analyzer": "active",
            "visualizer": "active"
        },
        "datasets_processed": len(processed_datasets),
        "analyses_completed": len(analysis_results)
    }

@app.post("/analyze-url")
async def analyze_url(url_data: Dict[str, str]):
    """Analyze datasets from a given URL"""
    start_time = datetime.now()
    try:
        url = url_data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        logger.info(f"Starting analysis for URL: {url}")
        logger.data_operation("URL_ANALYSIS_START", url, "Beginning dataset discovery")
        
        # Step 1: Discover datasets using Data Discovery Agent
        datasets = discovery_agent.discover_datasets(url)
        if not datasets:
            raise HTTPException(status_code=404, detail="No datasets found at the provided URL")
        
        # Step 2: Download and process datasets using Data Processing Agent
        processed_data = {}
        analysis_summary = {}
        
        for dataset in datasets[:4]:  # Limit to first 4 datasets as requested
            try:
                # Download dataset
                file_path = discovery_agent.download_dataset(dataset, 'data/raw')
                if file_path and os.path.exists(file_path):
                    # Process dataset
                    df, summary = processor_agent.process_dataset(file_path)
                    if df is not None:
                        dataset_name = dataset['name']
                        processed_data[dataset_name] = df
                        analysis_summary[dataset_name] = summary
                        
                        # Check if this dataset has multiple sheets
                        if processor_agent.has_multiple_sheets(dataset_name):
                            logger.info(f"Dataset {dataset_name} has multiple sheets")
                        
                        # Analyze dataset using Job Market Analyzer Agent
                        analysis = analyzer_agent.analyze_job_market_data(df, dataset_name)
                        if analysis:
                            analysis_results[dataset_name] = analysis
                            
                            # Create visualizations using Visualization Agent
                            charts = visualizer.create_dashboard(df, analysis)
                            visualization_charts[dataset_name] = charts
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset['name']}: {str(e)}")
                continue
        
        # Store processed data globally (store only the DataFrame, not the full object)
        for name, df in processed_data.items():
            # Clean the name to match what the frontend expects
            clean_name = name.replace('_', ' ').replace('[', '[').replace(']', ']')
            processed_datasets[clean_name] = df
        
        # Clean analysis results for JSON serialization
        sample_analysis = None
        if analysis_results:
            first_key = list(analysis_results.keys())[0]
            sample_analysis = clean_analysis_results(analysis_results[first_key])
        
        # Clean datasets list for JSON serialization
        cleaned_datasets = []
        for dataset in datasets:
            cleaned_dataset = {}
            for key, value in dataset.items():
                try:
                    cleaned_dataset[key] = convert_to_serializable(value)
                except:
                    cleaned_dataset[key] = str(value)
            cleaned_datasets.append(cleaned_dataset)
        
        # Log performance metrics
        duration = (datetime.now() - start_time).total_seconds()
        logger.performance("URL_ANALYSIS", duration, f"Processed {len(datasets)} datasets")
        logger.data_operation("URL_ANALYSIS_COMPLETE", url, f"Successfully processed {len(datasets)} datasets")
        
        # Return results
        return {
            "status": "success",
            "url": url,
            "datasets_discovered": len(datasets),
            "datasets": cleaned_datasets,
            "datasets_processed": len(processed_data),
            "processed_datasets": list(processed_data.keys()),
            "analysis_available": list(analysis_results.keys()),
            "sample_analysis": sample_analysis,
            "message": f"Successfully processed {len(processed_data)} datasets from ABS"
        }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error analyzing URL {url} after {duration:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/datasets")
async def get_datasets():
    """Get list of processed datasets"""
    return {
        "total_datasets": len(processed_datasets),
        "datasets": list(processed_datasets.keys()),
        "analysis_available": list(analysis_results.keys()),
        "multi_sheet_datasets": list(processor_agent.multi_sheet_data.keys()),
        "debug_info": {
            "processed_datasets_keys": list(processed_datasets.keys()),
            "multi_sheet_data_keys": list(processor_agent.multi_sheet_data.keys()),
            "analysis_results_keys": list(analysis_results.keys())
        }
    }

@app.get("/dataset/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get specific dataset information"""
    if dataset_name not in processed_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = processed_datasets[dataset_name]
        summary = processor_agent.get_data_summary(dataset_name)
        analysis = analysis_results.get(dataset_name, {})
        
        # Clean all data for JSON serialization
        cleaned_summary = clean_analysis_results(summary) if summary else {}
        cleaned_analysis = clean_analysis_results(analysis) if analysis else {}
        
        # Clean sample data more carefully
        sample_data = df.head(10)
        cleaned_sample_data = []
        for _, row in sample_data.iterrows():
            cleaned_row = {}
            for col, val in row.items():
                try:
                    # Handle pandas specific types
                    if pd.isna(val):
                        cleaned_row[str(col)] = None
                    elif isinstance(val, (pd.Timestamp, pd.NaT)):
                        cleaned_row[str(col)] = convert_to_serializable(val)
                    else:
                        cleaned_row[str(col)] = convert_to_serializable(val)
                except Exception as e:
                    # If all else fails, convert to string
                    cleaned_row[str(col)] = str(val) if val is not None else None
            cleaned_sample_data.append(cleaned_row)
        
        # Check for sheet information
        sheet_info = {}
        if dataset_name.lower().endswith(('.xlsx', '.xls')):
            try:
                import pandas as pd
                excel_file = pd.ExcelFile(f"data/{dataset_name}")
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    sheet_info = {
                        "total_sheets": len(sheet_names),
                        "sheet_names": sheet_names,
                        "has_multiple_sheets": True
                    }
                else:
                    sheet_info = {
                        "total_sheets": 1,
                        "sheet_names": sheet_names,
                        "has_multiple_sheets": False
                    }
            except Exception as e:
                sheet_info = {"error": f"Could not read sheet info: {str(e)}"}
        
        return {
            "dataset_name": dataset_name,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "summary": cleaned_summary,
            "analysis": cleaned_analysis,
            "sample_data": cleaned_sample_data,
            "sheet_info": sheet_info
        }
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset: {str(e)}")

@app.get("/analysis/{dataset_name}")
async def get_analysis(dataset_name: str):
    """Get analysis results for a specific dataset"""
    if dataset_name not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        cleaned_analysis = clean_analysis_results(analysis_results[dataset_name])
        return cleaned_analysis
    except Exception as e:
        logger.error(f"Error cleaning analysis for {dataset_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing analysis: {str(e)}")

@app.get("/sheets/{dataset_name}")
async def get_sheets(dataset_name: str):
    """Get all sheets for a specific dataset"""
    try:
        # Debug: Show what we're checking
        logger.info(f"Checking sheets for dataset: {dataset_name}")
        logger.info(f"Available multi-sheet datasets: {list(processor_agent.multi_sheet_data.keys())}")
        logger.info(f"Available processed datasets: {list(processed_datasets.keys())}")
        
        if processor_agent.has_multiple_sheets(dataset_name):
            logger.info(f"Dataset {dataset_name} has multiple sheets")
            sheets = processor_agent.get_all_sheets(dataset_name)
            sheet_info = {}
            for sheet_name, df in sheets.items():
                # Clean sample data for each sheet
                sample_data = df.head(10)
                cleaned_sample_data = []
                for _, row in sample_data.iterrows():
                    cleaned_row = {}
                    for col, val in row.items():
                        try:
                            if pd.isna(val):
                                cleaned_row[str(col)] = None
                            elif isinstance(val, (pd.Timestamp, pd.NaT)):
                                cleaned_row[str(col)] = convert_to_serializable(val)
                            else:
                                cleaned_row[str(col)] = convert_to_serializable(val)
                        except Exception as e:
                            cleaned_row[str(col)] = str(val) if val is not None else None
                    cleaned_sample_data.append(cleaned_row)
                
                sheet_info[sheet_name] = {
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "sample_data": cleaned_sample_data
                }
            
            return {
                "dataset_name": dataset_name,
                "has_multiple_sheets": True,
                "total_sheets": len(sheets),
                "sheet_names": list(sheets.keys()),
                "sheets": sheet_info
            }
        else:
            logger.info(f"Dataset {dataset_name} does not have multiple sheets")
            return {
                "dataset_name": dataset_name,
                "has_multiple_sheets": False,
                "total_sheets": 1,
                "sheet_names": [dataset_name],
                "sheets": {}
            }
    except Exception as e:
        logger.error(f"Error getting sheets for {dataset_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sheets: {str(e)}")

@app.get("/visualizations/{dataset_name}")
async def get_visualizations(dataset_name: str):
    """Get visualizations for a specific dataset"""
    if dataset_name not in visualization_charts:
        raise HTTPException(status_code=404, detail="Visualizations not found")
    
    charts = visualization_charts[dataset_name]
    # Convert Plotly figures to JSON-serializable format
    chart_data = {}
    for chart_name, fig in charts.items():
        chart_data[chart_name] = fig.to_dict()
    
    return chart_data

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and analyze a dataset file"""
    try:
        # Save uploaded file
        file_path = f"data/raw/{file.filename}"
        os.makedirs("data/raw", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the uploaded file
        df, summary = processor_agent.process_dataset(file_path)
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to process uploaded file")
        
        # Analyze the data
        analysis = analyzer_agent.analyze_job_market_data(df, file.filename)
        if not analysis:
            raise HTTPException(status_code=400, detail="Failed to analyze uploaded file")
        
        # Create visualizations
        charts = visualizer.create_dashboard(df, analysis)
        
        # Store results
        dataset_name = file.filename
        processed_datasets[dataset_name] = df
        analysis_results[dataset_name] = analysis
        visualization_charts[dataset_name] = charts
        
        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "dataset_name": dataset_name,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "analysis_available": True
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/export/{dataset_name}")
async def export_dataset(dataset_name: str, format: str = "csv"):
    """Export a processed dataset"""
    if dataset_name not in processed_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = processed_datasets[dataset_name]
    
    if format.lower() == "csv":
        csv_content = df.to_csv(index=False)
        return {"csv_data": csv_content}
    elif format.lower() == "json":
        json_data = df.to_dict('records')
        return {"json_data": json_data}
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'")

if __name__ == "__main__":
    # Create data directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/preprocessed", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    
    # Log startup information
    logger.info("Starting Job Market Analysis Agentic AI Backend")
    logger.info(f"Data directories: data/raw, data/preprocessed, charts")
    logger.info(f"Server will start on http://0.0.0.0:8000")
    logger.info(f"API Documentation: http://0.0.0.0:8000/docs")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
