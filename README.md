# Sri Lanka Bird Biodiversity - Ensemble Learning Dashboard

Student: Indrajith KTN 224078X

Predicts **recorded bird species richness** for Sri Lankan districts using ensemble learning
(Bagging, Boosting, Stacking) and hybrid models (RF + ANN, RF + ARIMA), with an interactive Streamlit dashboard.

> **AI use declaration:** The code in this repository (notebooks and `app.py`) was generated with the help of
> Claude (Anthropic). I ran the code, checked the outputs and wrote the interpretation myself.

## Live dashboard
[WRITE HERE: Streamlit Cloud link]

## Folder structure
```
app.py                     Streamlit dashboard
requirements.txt           packages for the dashboard
requirements-notebooks.txt extra packages for the notebooks
notebooks/                 step0 data preparation, step1 models, step2 time series, step3 explainability, step4 report
data/processed/            processed monthly dataset, results, predictions, SHAP image
models/                    trained models (joblib .pkl)
```
The raw dataset (about 420 MB) is not in this repository because of GitHub's size limit.
Source: https://huggingface.co/datasets/DilushaChandrasiri/SriLanka-Bird-Diversity-Dataset

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
To re-run the notebooks: `pip install -r requirements-notebooks.txt`, then run
step1 -> step2 -> step3 -> step4 in order.

## Results (test set 2023-2024)
- Tabular: best = Hybrid RF + ANN (average), MAE 25.43, R2 0.495
- Time series: best = SARIMA, MAE 11.51 (RF + ARIMA improves ARIMA; RF + SARIMA does not improve SARIMA)
- See `notebooks/step4_report.ipynb` for the full comparison and discussion.
