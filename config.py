"""
Configuration file for Job Market Analysis Agentic AI
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

STATIC_DIR = BASE_DIR / "static"

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

# API configuration
API_TITLE = "Job Market Analysis Agentic AI"
API_VERSION = "1.0.0"
API_DESCRIPTION = "An intelligent AI agent for analyzing job market datasets"

# Data processing configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json']
CHUNK_SIZE = 8192

# Analysis configuration
TREND_ANALYSIS_WINDOW = int(os.getenv("TREND_ANALYSIS_WINDOW", 10))
OUTLIER_THRESHOLD = float(os.getenv("OUTLIER_THRESHOLD", 1.5))
MIN_DATA_POINTS = int(os.getenv("MIN_DATA_POINTS", 5))


# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"

# ABS-specific configuration
ABS_BASE_URL = "https://www.abs.gov.au"
ABS_JOB_VACANCIES_URL = "https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia/may-2025"
ABS_DATASET_PATTERNS = [
    r'download.*xlsx',
    r'download.*csv',
    r'table.*download',
    r'data.*download'
]

# User agent for web scraping
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Request configuration
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", 3))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 1.0))

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 hour

# Security configuration
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Feature flags
ENABLE_HTML_EXTRACTION = os.getenv("ENABLE_HTML_EXTRACTION", "true").lower() == "true"
ENABLE_AUTO_ANALYSIS = os.getenv("ENABLE_AUTO_ANALYSIS", "true").lower() == "true"

def create_directories():
    """Create necessary directories if they don't exist"""
    for directory in [DATA_DIR, LOGS_DIR, STATIC_DIR]:
        directory.mkdir(exist_ok=True)

def get_config():
    """Get configuration as dictionary"""
    return {
        'base_dir': str(BASE_DIR),
        'data_dir': str(DATA_DIR),
        'logs_dir': str(LOGS_DIR),
        'host': HOST,
        'port': PORT,
        'streamlit_port': STREAMLIT_PORT,
        'api_title': API_TITLE,
        'api_version': API_VERSION,
        'max_file_size': MAX_FILE_SIZE,
        'supported_formats': SUPPORTED_FORMATS,
        'log_level': LOG_LEVEL,
        'abs_url': ABS_JOB_VACANCIES_URL,
        'cache_enabled': CACHE_ENABLED,
        'enable_auto_analysis': ENABLE_AUTO_ANALYSIS,
    }

# Create directories on import
create_directories()

