# Sydney & Melbourne Housing Price Analysis

This project investigates the macroeconomic forces driving house prices in 
Sydney and Melbourne using official ABS and RBA data. It forms the reproducible 
code base accompanying the report: **“What drives house prices? Evidence from 
macro indicators and dynamic responses.”**

---

## 🔍 Research Question
**Which macroeconomic factors are most strongly associated with short-run and 
long-run movements in Australian house prices?**

---

## 📂 Repository Structure
Housing-price-project/
├── data/ # cleaned panel data
├── notebooks/ # Jupyter notebook (full analysis)
├── report/ # final written report (PDF)
├── outputs/ # tables & figures generated from code
└── src/ # modular analysis code


---

## 📘 Notebook

All empirical results used in the report are fully reproducible via the notebook:

➡️ **`notebooks/housing_analysis.ipynb`**

This notebook includes:
- Data cleaning and transformation  
- Fixed-effects model (levels)  
- First-difference / DL model (short-run effects)  
- Coefficient tables  
- Figures used in the report  

---

## 🛠 To Reproduce the Results

```bash
pip install -r requirements.txt
python src/run_all.py

