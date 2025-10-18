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

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **Git** for cloning the repository

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Rabby3075/job-market-analysis-agenticai.git
   cd job-vacancies-agent
   ```

2. **Create Virtual Environment**:

   ```bash

   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
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

### ABS Industry Dashboard

1. **Start the application** using the Quick Start instructions above
2. **Click "📊 ABS Industry Dashboard"** on the landing page
3. **Automatic processing**:
   - Downloads latest ABS job vacancies data
   - Filters for industry-specific data (Table 4)
   - Preprocesses and cleans the data
   - Generates comprehensive analysis
4. **Explore the dashboard**:
   - View dataset preview and information
   - Analyze industry trends and patterns
   - Generate interactive visualizations
   - Access forecasting insights

### IVI (IT Jobs) Dashboard

1. **Click "💻 IVI (IT Jobs) Dashboard"** on the landing page
2. **Automatic processing**:
   - Downloads IVI ANZSCO4 occupation data
   - Filters for IT-related job categories
   - Preprocesses state-wise vacancy data
   - Performs specialized IT job market analysis
3. **Explore IT job insights**:
   - View IT job categories and distributions
   - Analyze state-wise IT job vacancies
   - Track IT job market trends over time
   - Generate IT-specific forecasting

### Key Features

- **🤖 Fully Automated**: No manual data entry required
- **📊 Real-time Analysis**: Fresh data from official sources
- **🎨 Interactive Visualizations**: Charts, graphs, and maps
- **🔮 Forecasting**: Predictive analytics for future trends
- **📈 Comprehensive Insights**: Trends, patterns, and recommendations

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
│   └── logger.py            # logging system
├── 📁 data/                 # Downloaded datasets
│   ├── raw/                 # Original datasets
│   └── preprocessed/        # Clean, processed data (CSV)

└── 📝 logs/                 #  log files
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
