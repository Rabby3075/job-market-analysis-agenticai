# 🔍 Job Market Analysis Agentic AI Application

An intelligent AI agent that automatically discovers, downloads, preprocesses, and analyzes job market datasets from various sources. Built specifically for analyzing the [ABS Job Vacancies data](https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia/may-2025) and other job market datasets.

## ✨ Features

- **🔍 URL-based Dataset Discovery**: Automatically finds and downloads datasets from provided URLs
- **📊 Multi-format Support**: Handles CSV, XLSX, XLS, JSON, and HTML tables
- **🧹 Intelligent Preprocessing**: Automatically cleans, validates, and prepares data for analysis
- **🧠 Advanced Analytics**: Provides comprehensive job market insights and trends
- **🌐 Modern Web Interface**: Built with FastAPI and Streamlit
- **🤖 Agentic AI**: Multiple specialized AI agents working together
- **📊 Universal CSV Output**: All preprocessed data saved as CSV regardless of original format
- **📝 Professional Logging**: Comprehensive logging system with file output and rotation

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **Git** for cloning the repository

### Option 1: One-Click Startup (Windows - Recommended)

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd job-vacancies-agent
   ```

2. **Double-click `start_app.bat`**:

   - This script will automatically:
     - Create a virtual environment
     - Install all dependencies
     - Create necessary directories
     - Start both backend and frontend servers
   - Two command windows will open (backend and frontend)

3. **Access the Application**:
   - 🌐 **FastAPI Backend**: http://localhost:8000
   - 📚 **API Docs**: http://localhost:8000/docs
   - 🎨 **Streamlit Frontend**: http://localhost:8501

### Option 2: Manual Setup

1. **Create Virtual Environment**:

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Create Data Directories**:

   ```bash
   mkdir data data\raw data\preprocessed logs charts
   ```

4. **Start the Application**:

   ```bash
   # Terminal 1: Start FastAPI Backend
   python main.py

   # Terminal 2: Start Streamlit Frontend
   streamlit run streamlit_app.py
   ```

5. **Access the Application**:
   - 🌐 **FastAPI Backend**: http://localhost:8000
   - 📚 **API Docs**: http://localhost:8000/docs
   - 🎨 **Streamlit Frontend**: http://localhost:8501

## 🎯 Usage

### For ABS Job Vacancies Data

1. **Enter the ABS URL**: `https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia/may-2025`
2. **Agent automatically discovers** the 4 key datasets:
   - Table 1: Job vacancies by states and territories
   - Table 2: Private sector job vacancies by states and territories
   - Table 3: Public sector job vacancies by states and territories
   - Table 4: Job vacancies by industry across Australia
3. **Downloads and preprocesses** the data automatically
4. **Analyzes** key metrics, trends, and insights

### For Other Datasets

1. **Provide any URL** containing job market data
2. **Upload your own files** (CSV, XLSX, XLS, JSON)
3. **Let the AI agents** handle the rest automatically

## 🏗️ Project Structure

```
job-vacancies-agent/
├── 🔧 main.py               # FastAPI backend
├── 🎨 streamlit_app.py      # Streamlit frontend
├── 🤖 agents/
│   ├── data_discovery.py    # URL scraping and dataset discovery
│   ├── data_processor.py    # Data preprocessing pipeline
│   └── analyzer.py          # Analysis and insights engine
├── 🛠️ utils/
│   └── logger.py            # Professional logging system
├── 📁 data/                 # Downloaded datasets
│   ├── raw/                 # Original datasets
│   └── preprocessed/        # Clean, processed data (CSV)

└── 📝 logs/                 # Professional log files
```

## 🤖 AI Agents Architecture

### 1. **Data Discovery Agent**

- Scrapes URLs to find downloadable datasets
- Identifies CSV, XLSX, and other data sources
- Extracts HTML tables when direct downloads aren't available

### 2. **Data Processing Agent**

- Cleans and validates raw data
- Handles missing values intelligently
- Optimizes data types and removes outliers
- Generates data quality reports

### 3. **Analysis Agent**

- Performs comprehensive job market analysis
- Identifies trends, patterns, and insights
- Generates actionable recommendations
- Analyzes geographic, industry, and sector distributions

## 📊 Supported Data Sources

- **Australian Bureau of Statistics (ABS)** - Primary focus
- **Government datasets** and reports
- **CSV files** and spreadsheets
- **Excel files** (XLSX, XLS)
- **JSON data** and APIs
- **HTML tables** from web pages

## 🔌 API Endpoints

### Core Endpoints

- `POST /analyze-url` - Analyze datasets from URL
- `POST /upload-dataset` - Upload and analyze file
- `GET /datasets` - List all processed datasets
- `GET /dataset/{name}` - Get specific dataset info
- `GET /analysis/{name}` - Get analysis results
- `GET /health` - System health check

### Example API Usage

```python
import requests

# Analyze ABS URL
response = requests.post("http://localhost:8000/analyze-url",
                        json={"url": "https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia/may-2025"})

# Get results
data = response.json()
print(f"Discovered {data['datasets_discovered']} datasets")
```

## 🎨 Frontend Options

### 1. **Streamlit Interface** (Recommended)

- Interactive data exploration
- User-friendly controls
- Access at: http://localhost:8501

### 2. **FastAPI Web Interface**

- Clean, modern design
- Direct API access
- Responsive layout
- Access at: http://localhost:8000

## 🔧 Configuration

### Environment Variables

Create a `.env` file for custom settings:

```env
DEBUG=True
LOG_LEVEL=INFO
MAX_FILE_SIZE=100MB
```

### Custom Analysis Rules

Modify `agents/analyzer.py` to add custom analysis logic for your specific use cases.

## 🚀 Deployment

### Local Development

For local development, follow the **Quick Start** instructions above. The application runs in development mode with:

- **Hot reload** enabled for both backend and frontend
- **Debug mode** activated
- **Detailed logging** for troubleshooting

### Production Deployment

1. **Set up production environment**:

   ```bash
   # Create production virtual environment
   python -m venv venv_prod
   source venv_prod/bin/activate  # Linux/Mac
   # or venv_prod\Scripts\activate  # Windows

   # Install production dependencies
   pip install -r requirements.txt
   ```

2. **Deploy with Gunicorn**:

   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

3. **Deploy with Docker**:
   ```bash
   docker build -t job-market-ai .
   docker run -p 8000:8000 job-market-ai
   ```

## 📈 Example Analysis Output

The system automatically generates:

- **📊 Overview Metrics**: Total vacancies, data quality, coverage
- **📈 Trend Analysis**: Growth patterns, seasonal variations
- **🗺️ Geographic Insights**: State/territory distributions
- **🏭 Industry Analysis**: Sector breakdowns and comparisons
- **💡 Key Insights**: Automated discovery of important patterns
- **🎯 Recommendations**: Actionable business intelligence

## 📊 Data Processing Pipeline

### **Overview**

The system automatically processes all raw data formats and converts them to standardized CSV format for consistency and analysis.

### **Raw Data Storage (`data/raw/`)**

- **Original files** in their native format (CSV, XLSX, XLS, JSON)
- **Preserved exactly** as downloaded or uploaded
- **No modifications** to original data

### **Preprocessed Data Storage (`data/preprocessed/`)**

- **All data converted to CSV** regardless of original format
- **Cleaned and standardized** column names and data types
- **Missing value handling** and outlier treatment
- **Data quality improvements** applied consistently

### **Multi-Sheet Excel Handling**

- **Each sheet processed individually** and saved as separate CSV
- **Sheet-specific folders** created for organization
- **Combined CSV file** with all sheets merged
- **Summary documentation** of all sheets and their properties

### **File Structure Example**

```
data/
├── raw/
│   ├── dataset1.xlsx          # Original Excel file
│   ├── dataset2.csv           # Original CSV file
│   └── dataset3.json          # Original JSON file
└── preprocessed/
    ├── dataset1_preprocessed.csv           # Main processed CSV
    ├── dataset1_all_sheets_combined.csv   # All sheets merged
    ├── dataset1_sheets/                   # Individual sheets
    │   ├── Sheet1.csv
    │   ├── Sheet2.csv
    │   └── _sheets_summary.txt
    ├── dataset2_preprocessed.csv           # Processed CSV
    └── dataset3_preprocessed.csv           # Processed CSV
```

## 📝 Professional Logging System

### **Overview**

The system includes a comprehensive logging framework that tracks all operations, performance metrics, and system events.

### **Log Files Structure**

```
logs/
├── main_all.log              # All logs from main application
├── main_errors.log           # Error logs only
├── main_performance.log      # Performance metrics
├── data_discovery_all.log    # Data discovery agent logs
├── data_processor_all.log    # Data processing agent logs
├── analyzer_all.log          # Analysis agent logs
```

### **Log Levels**

- **DEBUG**: Detailed debugging information
- **INFO**: General operational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical system failures

### **Specialized Logging**

- **Performance Logging**: Tracks execution times for operations
- **Data Operation Logging**: Records dataset processing activities
- **API Request Logging**: Monitors API endpoint usage and performance
- **Error Tracking**: Detailed error logging with context

### **Log Rotation**

- **File Size Limits**: 10MB for general logs, 5MB for error logs
- **Backup Count**: Keeps 5 backup files for general logs, 3 for error logs
- **Automatic Management**: Handles log file rotation automatically

### **Usage Example**

```python
from utils.logger import get_logger

# Get logger for your component
logger = get_logger("my_component")

# Log different types of information
logger.info("Operation started")
logger.performance("DATA_PROCESSING", 2.5, "Processed 1000 rows")
logger.error("Failed to process dataset")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Australian Bureau of Statistics for providing comprehensive job market data
- FastAPI and Streamlit communities for excellent frameworks
