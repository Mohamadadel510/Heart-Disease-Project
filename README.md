# Heart Disease Prediction ML Pipeline

## Project Overview
Complete machine learning pipeline for heart disease prediction using UCI dataset.

## Setup Instructions
1. Clone repository
2. Create virtual environment: `python -m venv heart_disease_env`
3. Activate environment: `source heart_disease_env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run notebooks in order (01-06)
6. Launch Streamlit app: `streamlit run ui/app.py`
7. Deploy with ngrok: `ngrok http 8501`

## Project Structure
- `notebooks/`: Jupyter notebooks for each step
- `models/`: Saved ML models and pipelines
- `ui/`: Streamlit web application
- `data/`: Dataset and processed data
- `results/`: Evaluation metrics and visualizations

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Means Clustering
- Hierarchical Clustering
```

## Execution Order

1. **Setup environment and download data**
2. **Run notebooks sequentially**: 01 → 02 → 03 → 04 → 05 → 06
3. **Test Streamlit app locally**: `streamlit run ui/app.py`
4. **Deploy with ngrok**: `ngrok http 8501`
5. **Push to GitHub**: Create repo and push all files

## Key Files to Generate
- All 6 Jupyter notebooks
- `final_tuned_model.pkl` and `complete_pipeline.pkl`
- `selected_features.pkl`
- Streamlit app (`ui/app.py`)
- `requirements.txt` and `README.md`