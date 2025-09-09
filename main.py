"""
Job Market Analysis Agentic AI - FastAPI Backend
Main application orchestrating the AI agents
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from agents.analyzer import JobMarketAnalyzer
# Import our AI agents
from agents.data_discovery import DataDiscoveryAgent
from agents.data_processor import DataProcessorAgent
from agents.visualizer_agent import JobMarketVisualizer
from utils.json_helper import clean_analysis_results, convert_to_serializable
from utils.logger import get_logger

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
visualizer_agent = JobMarketVisualizer()

# Global storage for processed data and results
processed_datasets = {}
analysis_results = {}
visualization_charts = {}

# Mount static files (if directory exists)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

def restore_from_disk() -> int:
    """Scan data/preprocessed for CSVs and load them into memory, rebuilding
    analyses and charts if missing. Returns number of datasets restored.
    """
    import glob
    count = 0
    # Load sheet CSVs inside *_sheets folders (prefer Data1.csv when exists)
    for folder in glob.glob("data/preprocessed/*_sheets"):
        # choose primary sheet if available, else any csv
        candidates = []
        data1 = os.path.join(folder, "Data1.csv")
        if os.path.exists(data1):
            candidates.append((os.path.basename(os.path.dirname(folder)), data1))
        else:
            for csv_path in glob.glob(os.path.join(folder, "*.csv")):
                candidates.append((os.path.basename(os.path.dirname(folder)), csv_path))
        for base_name, csv_path in candidates[:1]:
            try:
                df = pd.read_csv(csv_path)
                dataset_name = f"{base_name}.xlsx"
                processed_datasets[dataset_name] = df
                # Build analysis and charts if missing
                if dataset_name not in analysis_results:
                    analysis_results[dataset_name] = analyzer_agent.analyze_job_market_data(df, dataset_name)
                # Build visualizations
                try:
                    figs = visualizer_agent.create_dashboard(df, analysis_results.get(dataset_name, {}))
                    visualization_charts[dataset_name] = {k: json.loads(v.to_json()) for k, v in figs.items()}
                except Exception as e:
                    logger.warning(f"Visualization build failed for {dataset_name}: {e}")
                count += 1
            except Exception as e:
                logger.error(f"Failed to restore {csv_path}: {e}")
    # Also load flat CSVs directly under data/preprocessed
    for csv_path in glob.glob("data/preprocessed/*.csv"):
        try:
            df = pd.read_csv(csv_path)
            dataset_name = os.path.basename(csv_path).rsplit(".csv", 1)[0]
            processed_datasets[dataset_name] = df
            if dataset_name not in analysis_results:
                analysis_results[dataset_name] = analyzer_agent.analyze_job_market_data(df, dataset_name)
            try:
                figs = visualizer_agent.create_dashboard(df, analysis_results.get(dataset_name, {}))
                visualization_charts[dataset_name] = {k: json.loads(v.to_json()) for k, v in figs.items()}
            except Exception as e:
                logger.warning(f"Visualization build failed for {dataset_name}: {e}")
            count += 1
        except Exception as e:
            logger.error(f"Failed to restore {csv_path}: {e}")
    return count

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

@app.post("/restore-datasets")
async def restore_datasets():
    """Rescan disk and restore datasets/analyses/charts to memory."""
    restored = restore_from_disk()
    return {"restored": restored, "datasets": list(processed_datasets.keys())}

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
                            try:
                                figs = visualizer_agent.create_dashboard(df, analysis)
                                visualization_charts[dataset['name']] = {k: json.loads(v.to_json()) for k, v in figs.items()}
                            except Exception as e:
                                logger.warning(f"Visualization build failed during /analyze-url for {dataset['name']}: {e}")
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset['name']}: {str(e)}")
                continue
        
        # Store processed data globally (store only the DataFrame, not the full object)
        for name, df in processed_data.items():
            # Clean the name to match what the frontend expects
            clean_name = name.replace('_', ' ').replace('[', '[').replace(']', ']')
            processed_datasets[clean_name] = df
            # Also re-key analysis and visualizations to the cleaned name for consistent access
            if name in analysis_results:
                analysis_results[clean_name] = analysis_results.pop(name)
            if name in visualization_charts:
                visualization_charts[clean_name] = visualization_charts.pop(name)
        
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

@app.post("/abs/download")
async def abs_download(command: Dict[str, str]):
    """Download ABS Job Vacancies release by natural command.
    Body: { "query": "latest" | "May 2025" | "February 2024" }
    Downloads raw files only and returns file paths. No preprocessing here.
    """
    abs_root = command.get("root_url") or "https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia"
    query = command.get("query") or "latest"
    try:
        # Resolve all releases and latest
        releases = discovery_agent.resolve_abs_release_links(abs_root)
        if not releases:
            raise HTTPException(status_code=404, detail="No releases found on ABS page")
        latest = max(releases, key=lambda r: (r['year'], r['month']))
        intent = discovery_agent.parse_release_query(query)

        # Handle future requests beyond latest
        if not intent['latest'] and intent['month'] is not None and intent['year'] is not None:
            if (intent['year'], intent['month']) > (latest['year'], latest['month']):
                return {
                    "status": "error",
                    "error_type": "future_release",
                    "message": f"Requested release {query} not published yet. Latest is {latest['year']}-{latest['month']:02d}.",
                    "latest": {"month": latest['month'], "year": latest['year'], "title": latest['title']}
                }

        # Find target
        target = None
        if intent['latest']:
            target = latest
        else:
            # exact match
            for r in releases:
                if intent['month'] == r['month'] and (intent['year'] is None or intent['year'] == r['year']):
                    if intent['year'] is None:
                        # pick most recent for that month
                        if (target is None) or (r['year'], r['month']) > (target['year'], target['month']):
                            target = r
                    else:
                        target = r
                        break
            if target is None:
                # compute nearest by absolute difference in months
                if intent['month'] is not None:
                    import math
                    requested_year = intent['year'] if intent['year'] is not None else latest['year']
                    requested = requested_year * 12 + intent['month']
                    nearest = min(releases, key=lambda r: abs((r['year'] * 12 + r['month']) - requested))
                    # Return suggestion without downloading
                    return {
                        "status": "not_available",
                        "message": f"Requested release {query} is not available.",
                        "suggestion": {"month": nearest['month'], "year": nearest['year'], "title": nearest['title']}
                    }

        # If we have a target, download
        paths = discovery_agent.discover_datasets(target['url'])
        downloaded_paths = []
        for ds in paths[:4]:
            file_path = discovery_agent.download_dataset(ds, 'data/raw')
            if file_path:
                downloaded_paths.append(file_path)
        if not downloaded_paths:
            raise HTTPException(status_code=404, detail="No files downloaded for the specified release")
        return {"status": "success", "downloaded": downloaded_paths, "query": query, "resolved": {"month": target['month'], "year": target['year']}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ABS download failed for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"ABS download failed: {str(e)}")

@app.post("/process-files")
async def process_files(payload: Dict[str, List[str]]):
    """Process a list of local dataset file paths, analyze, visualize, and register them in-memory.
    Body: { "paths": ["data/raw/file1.xlsx", ...] }
    """
    paths = payload.get("paths", [])
    if not paths:
        raise HTTPException(status_code=400, detail="No file paths provided")
    processed = []
    for file_path in paths:
        try:
            # Normalize and try fallbacks if the file was renamed or URL-encoded
            orig = file_path
            try:
                import urllib.parse
                file_path = urllib.parse.unquote(file_path)
            except Exception:
                pass
            file_path = file_path.replace("\\", "/")
            if not os.path.exists(file_path):
                # Try basename under data/raw
                base_try = os.path.join("data/raw", os.path.basename(file_path))
                if os.path.exists(base_try):
                    file_path = base_try
                else:
                    # Try ABS canonical renamed files with similar stem
                    import glob
                    stem = os.path.splitext(os.path.basename(file_path))[0]
                    candidates = glob.glob("data/raw/ABS_*.xlsx")
                    if candidates:
                        # Use the most recent ABS file as a fallback
                        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        file_path = candidates[0]
            if not os.path.exists(file_path):
                logger.error(f"File not found for processing (after fallbacks): {orig}")
                continue
            df, summary = processor_agent.process_dataset(file_path)
            if df is None:
                continue
            raw_name = os.path.basename(file_path)
            clean_name = raw_name.replace('_', ' ').replace('[', '[').replace(']', ']')
            processed_datasets[clean_name] = df
            analysis = analyzer_agent.analyze_job_market_data(df, clean_name)
            analysis_results[clean_name] = analysis or {}
            # Build visualizations
            try:
                figs = visualizer_agent.create_dashboard(df, analysis)
                visualization_charts[clean_name] = {k: json.loads(v.to_json()) for k, v in figs.items()}
            except Exception as e:
                logger.warning(f"Visualization build failed for {clean_name}: {e}")
            processed.append({
                "dataset_name": clean_name,
                "shape": list(df.shape),
                "columns": list(df.columns)
            })
        except Exception as e:
            logger.error(f"Failed processing {file_path}: {e}")
            continue
    if not processed:
        raise HTTPException(status_code=400, detail="No files were processed")
    return {"status": "success", "processed": processed, "count": len(processed)}

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
async def get_visualizations(dataset_name: str, mode: str = "auto"):
    """Get visualizations for a specific dataset"""
    # Tolerant lookup by several name variants
    candidates = [
        dataset_name,
        dataset_name.replace(' ', '_'),
        dataset_name.replace('_', ' '),
        dataset_name.replace(' ', ''),
    ]
    charts = None
    for key in visualization_charts.keys():
        if key in candidates or key.replace('_', ' ') in candidates:
            charts = visualization_charts[key]
            break
    if charts is None or not charts:
        # Try to build on the fly from processed dataset if available
        df = processed_datasets.get(dataset_name)
        if df is None:
            # try alternative keys
            for key in processed_datasets.keys():
                if key in candidates or key.replace('_', ' ') in candidates:
                    df = processed_datasets[key]
                    break
        if df is None:
            raise HTTPException(status_code=404, detail="Visualizations not available for this dataset")
        try:
            if mode == "notebook":
                figs = visualizer_agent.create_dashboard_from_notebook(df, "notebook/visualization.ipynb")
            else:
                figs = visualizer_agent.create_dashboard(df, analysis_results.get(dataset_name, {}))
            charts = {k: json.loads(v.to_json()) for k, v in figs.items()}
            visualization_charts[dataset_name] = charts
        except Exception as e:
            logger.error(f"Failed to build visualizations for {dataset_name}: {e}")
            raise HTTPException(status_code=500, detail="Failed to build visualizations")
    return charts

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
        
        # Store results
        dataset_name = file.filename
        processed_datasets[dataset_name] = df
        analysis_results[dataset_name] = analysis
        # Visualization disabled
        
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
    # Restore any previously saved datasets
    restored = restore_from_disk()
    logger.info(f"Restored {restored} datasets from disk")
    
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
