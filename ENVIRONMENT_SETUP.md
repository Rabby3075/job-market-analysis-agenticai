# 🌍 Environment Setup Guide

## 🎯 **Why Virtual Environments Are CRITICAL**

### **❌ Problems WITHOUT Virtual Environments:**
- **Package conflicts** between projects
- **System Python pollution** - breaks other applications
- **Version mismatches** - "it works on my machine"
- **Unprofessional** - shows lack of software engineering knowledge
- **Capstone failure risk** - professors expect this

### **✅ Benefits WITH Virtual Environments:**
- **Isolated dependencies** - each project has its own packages
- **Reproducible builds** - exact same environment every time
- **Professional practice** - industry standard
- **Easy cleanup** - delete environment, start fresh
- **Capstone success** - shows you understand best practices

## 🐍 **Option 1: Python venv (Recommended)**

### **Step 1: Create Environment**
```bash
# Navigate to your project directory
cd E:\Project\job-vacancies-agent

# Create virtual environment
python -m venv job_analysis_env
```

### **Step 2: Activate Environment**
```bash
# Windows (Command Prompt)
job_analysis_env\Scripts\activate.bat

# Windows (PowerShell)
job_analysis_env\Scripts\Activate.ps1

# Linux/Mac
source job_analysis_env/bin/activate
```

### **Step 3: Install Dependencies**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt
```

### **Step 4: Verify Installation**
```bash
# Check Python location (should show your environment)
where python

# Check installed packages
pip list
```

## 🐍 **Option 2: Conda (Alternative)**

### **Step 1: Install Anaconda/Miniconda**
- Download from: https://docs.conda.io/en/latest/miniconda.html
- Install with default settings

### **Step 2: Create Environment**
```bash
# Create conda environment
conda create -n job_analysis python=3.9

# Activate environment
conda activate job_analysis
```

### **Step 3: Install Dependencies**
```bash
# Install packages
conda install pandas numpy plotly
pip install fastapi uvicorn streamlit
```

## 🚀 **Daily Workflow**

### **Starting Work:**
```bash
# 1. Navigate to project
cd E:\Project\job-vacancies-agent

# 2. Activate environment
job_analysis_env\Scripts\activate.bat  # Windows
# OR
conda activate job_analysis            # Conda

# 3. Verify activation (should see environment name in prompt)
# (job_analysis_env) E:\Project\job-vacancies-agent>
```

### **Running Your Project:**
```bash
# Terminal 1: Backend
python main.py

# Terminal 2: Frontend
streamlit run streamlit_app.py
```

### **Stopping Work:**
```bash
# Deactivate environment
deactivate  # venv
# OR
conda deactivate  # conda
```

## 📦 **Managing Dependencies**

### **Adding New Packages:**
```bash
# Install new package
pip install new_package

# Update requirements.txt
pip freeze > requirements.txt
```

### **Updating Requirements:**
```bash
# Install exact versions from requirements.txt
pip install -r requirements.txt
```

### **Clean Install:**
```bash
# Remove environment and recreate
rmdir /s job_analysis_env
python -m venv job_analysis_env
job_analysis_env\Scripts\activate.bat
pip install -r requirements.txt
```

## 🔧 **Troubleshooting**

### **Common Issues:**

#### **1. "pip not found"**
```bash
# Solution: Upgrade pip in environment
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

#### **2. "Activate script not found"**
```bash
# Use full path or recreate environment
python -m venv job_analysis_env --clear
```

#### **3. "Package conflicts"**
```bash
# Solution: Use virtual environment (isolates packages)
# Never install globally with pip
```

#### **4. "Different Python versions"**
```bash
# Check Python version in environment
python --version

# Create environment with specific version
python3.9 -m venv job_analysis_env
```

## 📚 **Best Practices**

### **✅ DO:**
- **Always use** virtual environments for projects
- **Activate environment** before working
- **Keep requirements.txt** updated
- **Use descriptive** environment names
- **Document** setup steps

### **❌ DON'T:**
- **Install packages globally** with pip
- **Mix different Python versions** in same environment
- **Forget to activate** environment
- **Delete requirements.txt**
- **Use system Python** for development

## 🎓 **For Your Capstone Project**

### **Why This Matters:**
1. **Professional appearance** - shows software engineering knowledge
2. **Reproducible results** - tutors can run your code exactly
3. **No conflicts** - your project won't break other systems
4. **Industry standard** - what companies expect
5. **Easy deployment** - can recreate environment anywhere

### **What Professors Look For:**
- ✅ **Virtual environment setup**
- ✅ **requirements.txt** with exact versions
- ✅ **Clear setup instructions**
- ✅ **Professional project structure**
- ✅ **No global package pollution**

## 🚀 **Quick Start Commands**

```bash
# 1. Create environment
python -m venv job_analysis_env

# 2. Activate (Windows)
job_analysis_env\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run project
python main.py
# In another terminal:
streamlit run streamlit_app.py
```

## 📝 **Environment Variables (Optional)**

Create `.env` file for configuration:
```bash
# .env file
HOST=0.0.0.0
PORT=8000
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
```

---

**🎯 Remember: Virtual environments are NOT optional for professional projects!**
**They're the foundation of modern Python development and essential for your capstone success.**
