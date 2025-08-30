# 📁 Data Folder Structure

## 🎯 **Overview**

Your agentic AI system now has a professional, organized data structure that separates raw and preprocessed data.

## 📂 **Folder Structure**

```
data/
├── raw/                    # 🔴 Raw datasets (original format)
│   ├── Downloadxlsx[70.31_KB].xlsx
│   ├── Downloadxlsx[82.31_KB].xlsx
│   ├── Downloadxlsx[93.67_KB].xlsx
│   ├── Downloadxlsx[84.62_KB].xlsx
│   ├── sample_job_vacancies_by_state.csv
│   ├── sample_job_vacancies_by_industry.csv
│   ├── sample_job_vacancies_time_series.csv
│   └── sample_job_vacancies_by_region.csv
│
├── preprocessed/           # 🟢 Clean, processed datasets (CSV format)
│   ├── Downloadxlsx[70.31_KB]_preprocessed.csv
│   ├── Downloadxlsx[82.31_KB]_preprocessed.csv
│   ├── Downloadxlsx[93.67_KB]_preprocessed.csv
│   ├── Downloadxlsx[84.62_KB]_preprocessed.csv
│   ├── sample_job_vacancies_by_state_preprocessed.csv
│   ├── sample_job_vacancies_by_industry_preprocessed.csv
│   ├── sample_job_vacancies_time_series_preprocessed.csv
│   └── sample_job_vacancies_by_region_preprocessed.csv
│
└── charts/                 # 📊 Generated visualizations
    ├── sample_job_vacancies_by_state/
    ├── sample_job_vacancies_by_industry/
    ├── sample_job_vacancies_time_series/
    └── sample_job_vacancies_by_region/
```

## 🔄 **Data Pipeline**

### **1. Raw Data (data/raw/)**

- **Source:** URLs, file uploads, demo data
- **Format:** Original format (XLSX, CSV, JSON, XLS)
- **Purpose:** Preserve original data integrity
- **Naming:** `filename.extension` or `Downloadxlsx[size].xlsx`

### **2. Preprocessed Data (data/preprocessed/)**

- **Source:** Processed from raw data
- **Format:** Always CSV (standardized)
- **Purpose:** Clean, ready-to-analyze data
- **Naming:** `filename_preprocessed.csv`

### **3. Multi-Sheet Excel Files**

For Excel files with multiple sheets, the system creates:

```
data/preprocessed/
└── Downloadxlsx[size]_sheets/
    ├── Index.csv
    ├── Data1.csv
    └── Enquiries.csv
```

## 🚀 **How It Works**

### **Data Discovery Agent:**

1. ✅ Scrapes URLs for datasets
2. ✅ Downloads to `data/raw/` folder
3. ✅ Preserves original format

### **Data Processing Agent:**

1. ✅ Loads raw data from `data/raw/`
2. ✅ Cleans and preprocesses data
3. ✅ Saves cleaned data to `data/preprocessed/` as CSV
4. ✅ Handles multiple sheets separately

### **Analysis & Visualization:**

1. ✅ Works with preprocessed data
2. ✅ Generates charts in `charts/` folder
3. ✅ Provides insights and trends

## 💡 **Benefits**

- **🧹 Clean Organization:** Raw vs. processed data clearly separated
- **🔄 Reproducibility:** Can reprocess raw data anytime
- **📊 Standardization:** All preprocessed data in CSV format
- **🔍 Easy Access:** Clear folder structure for different data types
- **💾 Version Control:** Keep original data while working with clean versions

## 🎯 **Usage Examples**

### **For Your Tutor:**

_"I built a multi-agent AI system with a professional data pipeline. Raw datasets are automatically downloaded to `data/raw/`, then my AI agents clean and preprocess them, saving the results as standardized CSV files in `data/preprocessed/`. This ensures data integrity while providing clean, analysis-ready datasets."_

### **For Development:**

- **Raw data:** `data/raw/` - Original files for reference
- **Clean data:** `data/preprocessed/` - Ready for analysis
- **Charts:** `charts/` - Generated visualizations

## 🔧 **Technical Details**

- **Automatic Creation:** Folders created on startup
- **File Naming:** Consistent naming convention
- **Format Conversion:** Raw → CSV automatically
- **Multi-sheet Support:** Excel sheets saved separately
- **Error Handling:** Graceful fallbacks if issues occur

This structure follows industry best practices for data science and machine learning projects! 🎉
