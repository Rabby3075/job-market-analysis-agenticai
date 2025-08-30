# 🚀 Quick Start Guide - Job Market Analysis AI

## ⚡ Get Started in 3 Simple Steps

### 1. 🎯 **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. 🚀 **Start the Backend (FastAPI)**

```bash
python main.py
```

This starts your AI backend at: http://localhost:8000

### 3. 🎨 **Start the Frontend (Streamlit)**

In a new terminal:

```bash
streamlit run streamlit_app.py
```

This opens your AI interface at: http://localhost:8501

## 🌐 **Access Your Application**

- **Main App**: http://localhost:8000
- **Streamlit**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 🔍 **Analyze ABS Job Vacancies Data**

1. Go to http://localhost:8501
2. Enter: `https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia/may-2025`
3. Click "🔍 Discover & Analyze"
4. Watch the AI agents work automatically!

## 🎯 **What You'll Get**

The system automatically discovers and analyzes the **4 key ABS datasets**:

1. **📊 Table 1**: Job vacancies by states and territories
2. **🏢 Table 2**: Private sector job vacancies by states and territories
3. **🏛️ Table 3**: Public sector job vacancies by states and territories
4. **🏭 Table 4**: Job vacancies by industry across Australia

## 🚨 **Troubleshooting**

### If you get dependency errors:

```bash
pip install -r requirements.txt
```

### If ports are busy:

- Change ports in `config.py`
- Or kill processes on ports 8000 and 8501

### If you want to test the API directly:

- Visit http://localhost:8000/docs for interactive API testing
- Use the FastAPI interface at http://localhost:8000

## 🎉 **You're Ready!**

The AI agents will automatically:

- 🔍 **Discover** datasets from URLs
- 📥 **Download** and extract data
- 🧹 **Clean** and preprocess data
- 🧠 **Analyze** trends and patterns
- 📈 **Visualize** insights interactively
- 💡 **Generate** actionable recommendations

## 📱 **Mobile Friendly**

Both interfaces work great on mobile devices for on-the-go analysis!

---

**Need help?** Check the full README.md for detailed documentation.
