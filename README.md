# Capstone6980
# Iowa Child Abuse Forecast — Streamlit Dashboard

Interactive dashboard accompanying the ALY 6980 Capstone XN project sponsored by Kids At Risk Action (KARA). Presents the full ten-week analytical arc (descriptive EDA, Phase 1 count modeling, Phase 2 rate-per-1,000 modeling, SHAP interpretability, spatial ablation, and the final 2024 forecast for all 99 Iowa counties).

## Contents

```
iowa_dashboard/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies (streamlit + pandas)
├── README.md           # This file
└── figures/            # 12 PNG figures embedded in the dashboard
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The dashboard intentionally has minimal dependencies. It does not retrain the models — model metrics, SHAP rankings, and forecast values are all encoded as cached DataFrames in `app.py`. The full retraining pipeline lives in the accompanying Jupyter notebook.

### 2. Launch the dashboard

From inside the `iowa_dashboard/` directory:

```bash
streamlit run app.py
```

Streamlit will open the dashboard in your default browser (typically http://localhost:8501). On first launch this can take a few seconds.

### 3. Navigate

The sidebar on the left contains the page navigator. Eight pages are available:

1. **Overview** — Project summary, headline metrics, and the model-evolution synthesis table.
2. **Descriptive EDA** — Phase 1 EDA findings (Figures 1–7) with key takeaways.
3. **Phase 1: Count Model** — The misleading-by-construction R² = 0.97 result and why it doesn't survive scrutiny.
4. **Phase 2: Rate Model** — The honest baseline (R² = 0.342) and the count-vs-rate comparison.
5. **SHAP Interpretability** — Per-feature contributions in the final model (Figures 10–11).
6. **Spatial Ablation** — The Week 10 like-for-like ablation that falsified the spatial hypothesis.
7. **Final 2024 Forecast** — Top counties by rate and by count, with statewide totals.
8. **County Detail Lookup** — Interactive single-county view with comparison to the statewide rate.

## Methodology summary

The dashboard summarizes work documented in detail in:

- The capstone final report (`ALY6980CAPSTONE_FinalReport.docx`).
- The Jupyter notebook (`ALY6980_FinalNotebook.ipynb`) containing all 38 numbered code blocks.
- The Week 10 biweekly report (`ALY6980CAPSTONE_BiWeeklyReport_M10.docx`).

The final model is XGBoost (400 trees, depth 5, learning rate 0.05) trained on a 990-row 99-county panel for fiscal years 2014–2023, predicting the rate of reported child abuse occurrences per 1,000 children under 18 (ACS B09001_001E). Final feature set: rate_lag1, rate_lag2, rate_lag3, pct_neglect, pct_physical, pct_sexual, pct_substance, Year, Under18, and one-hot-encoded County and Service Area fixed effects.

Final rolling-origin mean performance (2018–2023, six hold-out years): MAE 7.39, RMSE 9.91, R² 0.339.

## Data sources

- Iowa Department of Health and Human Services. (2024). *Child abuse occurrences by year, county, and type of abuse* [Data set]. State of Iowa Open Data Portal. https://data.iowa.gov
- U.S. Census Bureau. (2024). *American Community Survey 5-year estimates, Table B09001: Population under 18 years* [Data set]. https://api.census.gov/data.html

## Contact

Sreekarteek Akshinthala
Northeastern University, College of Professional Studies
ALY 6980 Capstone, Spring 2026
Instructor: Prof. Ajit Appari
Sponsor: Kids At Risk Action (KARA)
