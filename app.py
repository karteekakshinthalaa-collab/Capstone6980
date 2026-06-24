"""
Iowa Child Abuse Occurrences Dashboard
======================================
Streamlit dashboard for the ALY 6980 Capstone XN project sponsored by
Kids At Risk Action (KARA). Presents the full ten-week analytical arc
(EDA, Phase 1 count modeling, Phase 2 rate modeling, SHAP interpretability,
spatial ablation, final 2024 forecast) as an interactive multi-page
dashboard.
How to run
----------
    pip install -r requirements.txt
    streamlit run app.py
Then open the URL Streamlit prints (typically http://localhost:8501).
Submitted by: Sreekarteek Akshinthala
Instructor:   Prof. Ajit Appari
Date:         06/18/2026
"""
import os
import pandas as pd
import streamlit as st
# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Iowa Child Abuse Forecast — KARA Capstone",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Figures live in the same directory as app.py (repo root on Streamlit Cloud).
FIG_DIR = os.path.dirname(__file__)
# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------
@st.cache_data
def load_forecast_data():
    """Final 2024 forecast for all 99 Iowa counties (subset showing top 25)."""
    return pd.DataFrame(
        [
            ("Polk",          "5-Des Moines",   122384, 2325, 2228, 18.2),
            ("Linn",          "4-Cedar Rapids",  52233, 1242, 1260, 24.1),
            ("Scott",         "3-Eastern",       40786, 1184, 1110, 27.2),
            ("Dubuque",       "3-Eastern",       22431,  721,  739, 33.0),
            ("Woodbury",      "1-Western",       27420,  720,  739, 27.0),
            ("Pottawattamie", "1-Western",       21645,  691,  702, 32.4),
            ("Black Hawk",    "2-Northern",      28679,  503,  451, 15.7),
            ("Johnson",       "4-Cedar Rapids",  30276,  411,  468, 15.5),
            ("Des Moines",    "3-Eastern",        8444,  457,  334, 39.6),
            ("Lee",           "3-Eastern",        7075,  327,  341, 48.2),
            ("Muscatine",     "3-Eastern",       10344,  362,  337, 32.6),
            ("Webster",       "2-Northern",       7872,  348,  308, 39.2),
            ("Wapello",       "5-Des Moines",     8349,  323,  303, 36.3),
            ("Marshall",      "2-Northern",       9728,  280,  295, 30.3),
            ("Cerro Gordo",   "2-Northern",       8431,  236,  220, 26.1),
            ("Story",         "5-Des Moines",    16245,  220,  240, 14.8),
            ("Floyd",         "2-Northern",       3508,  172,  141, 40.3),
            ("Fayette",       "3-Eastern",        4145,  172,  149, 36.0),
            ("Tama",          "4-Cedar Rapids",   4079,  110,  159, 38.9),
            ("Boone",         "5-Des Moines",     5840,  140,  130, 22.3),
            ("Buchanan",      "3-Eastern",        5012,  142,  135, 26.9),
            ("Audubon",       "1-Western",        1249,  114,   84, 67.1),
            ("Greene",        "5-Des Moines",     1844,  105,   80, 43.5),
            ("Emmet",         "1-Western",        1817,   92,  104, 57.4),
            ("Osceola",       "1-Western",        1436,   73,   61, 42.5),
        ],
        columns=["County", "ServiceArea", "Under18_2023", "Actual_2023",
                 "Forecast_2024", "Forecast_Rate_2024"],
    )
@st.cache_data
def load_model_metrics():
    """Rolling-origin mean performance across all four model phases."""
    return pd.DataFrame(
        [
            ("Phase 1 — XGBoost Count",         "Wks 1–4",  "1,683 / 99",
             "34.21", "54.66", "0.970", "Inflated by population-size signal"),
            ("Phase 2 — XGBoost Rate",          "Wks 7–8",  "1,078 / 98",
             "7.36",  "9.89",  "0.342", "Honest baseline; pre_2014 dropped"),
            ("Phase 3 (Wk 9) — XGB + spatial",  "Wk 9",     "990 / 99",
             "7.54",  "10.05", "0.318", "Spatial added; comparison confounded"),
            ("Final — XGBoost Rate (no spatial)","Wk 10",    "990 / 99",
             "7.39",  "9.91",  "0.339", "Final model; spatial dropped via ablation"),
        ],
        columns=["Model", "Period", "Panel (rows / counties)",
                 "Mean MAE", "Mean RMSE", "Mean R²", "Notes"],
    )
@st.cache_data
def load_shap_importance():
    """Mean absolute SHAP value per feature in the final model."""
    return pd.DataFrame(
        [
            ("rate_lag1",                              3.60),
            ("Year",                                    2.08),
            ("pct_substance",                           1.63),
            ("County (fixed effects, summed)",          1.60),
            ("rate_lag3",                               1.34),
            ("rate_lag2",                               1.28),
            ("pct_physical",                            1.24),
            ("pct_sexual",                              1.05),
            ("Under18",                                 0.94),
            ("Service Area (fixed effects, summed)",    0.75),
            ("pct_neglect",                             0.59),
        ],
        columns=["Feature", "Mean |SHAP value|"],
    )
# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Iowa Child Abuse Forecast")
st.sidebar.markdown("*ALY 6980 Capstone • KARA*")
st.sidebar.markdown("---")
PAGES = [
    "Overview",
    "Descriptive EDA",
    "Phase 1: Count Model",
    "Phase 2: Rate Model",
    "SHAP Interpretability",
    "Spatial Ablation",
    "Final 2024 Forecast",
    "County Detail Lookup",
]
page = st.sidebar.radio("Navigate", PAGES, index=0)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sreekarteek Akshinthala**  \nNortheastern Univ. CPS  \nProf. Ajit Appari"
)
# ---------------------------------------------------------------------------
# Helper to display a captioned figure
# ---------------------------------------------------------------------------
def show_figure(filename, caption, width=None):
    path = os.path.join(FIG_DIR, filename)
    if os.path.exists(path):
        if width:
            st.image(path, caption=caption, width=width)
        else:
            st.image(path, caption=caption, use_column_width=True)
    else:
        st.warning(f"Figure not found: {path}")
# ===========================================================================
# PAGE 1 — Overview
# ===========================================================================
if page == "Overview":
    st.title("Iowa Child Abuse Occurrences — Predictive Model and 2024 Forecast")
    st.markdown("**Sponsor: Kids At Risk Action (KARA)**  •  Capstone XN Project")
    st.markdown("""
This dashboard presents the full ten-week analytical arc of the ALY 6980 capstone XN
project, predicting county-level reported child abuse occurrences in Iowa using publicly
available data. The analysis evolved through three deliberate methodological phases —
described in detail in the section pages — and produces a year-ahead forecast for all
99 Iowa counties as the primary deliverable for KARA's policy and advocacy work.
""")
    col1, col2, col3 = st.columns(3)
    col1.metric("Counties modeled",        "99 / 99")
    col2.metric("Statewide 2024 forecast", "16,908",
                delta="−1.6% vs 2023 actual (17,188)")
    col3.metric("Final model R² (rate)",   "0.339",
                delta="vs naive baseline 0.141")
    st.markdown("---")
    st.subheader("Model Evolution Across the Capstone")
    st.markdown(
        "Phase 1's headline R² of 0.97 was substantially inflated by free "
        "population-size signal. Phase 2's per-capita reformulation produced an "
        "honest baseline. Phase 3's experimental spatial term was cleanly falsified "
        "by Week 10's like-for-like ablation. The **Final** row is the model that "
        "produces the 2024 forecast on the next page."
    )
    st.dataframe(load_model_metrics(), use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("Project Phase Summary")
    cols = st.columns(4)
    cols[0].markdown("**Phase 1**  \n*Weeks 1–4*  \nEDA + count modeling")
    cols[1].markdown("**Phase 2**  \n*Weeks 7–8*  \nACS merge + rate model + SHAP")
    cols[2].markdown("**Phase 3**  \n*Week 9*  \nSpatial experiment + 99-cnty fix")
    cols[3].markdown("**Final**  \n*Week 10*  \nAblation + final SHAP + 2024 forecast")
# ===========================================================================
# PAGE 2 — Descriptive EDA
# ===========================================================================
elif page == "Descriptive EDA":
    st.title("Phase 1 — Descriptive Exploratory Analysis")
    st.markdown("Seven figures characterize the dataset before any modeling. "
                "All figures use the cleaned 9,611-row dataset spanning 2004–2023 "
                "across 99 counties and 14 abuse categories.")
    st.markdown("### Figure 1. Statewide Trend")
    show_figure("fig01_statewide_trend.png",
                "Figure 1. Statewide Reported Child Abuse Occurrences in Iowa, 2004–2023.")
    st.info("**Finding.** The mid-2010s drop (peak 22,120 in 2006 → low 10,830 in "
            "2014) coincides with Iowa DHS's documented reporting-system transition, "
            "not a true decline in incidence.")
    st.markdown("### Figure 2. Top 10 Counties by Volume")
    show_figure("fig02_top10_counties.png",
                "Figure 2. Top 10 Iowa Counties by Total Reported Child Abuse Occurrences, 2004–2023.")
    st.info("**Finding.** Reporting volume tracks population density. Polk County "
            "alone (45,428 occurrences) and four other metropolitan counties "
            "account for more than half of Iowa's twenty-year reported cases.")
    st.markdown("### Figure 3. Abuse Type Trends")
    show_figure("fig03_top5_types.png",
                "Figure 3. Annual Trend of the Five Most Common Abuse Types in Iowa, 2004–2023.")
    st.info("**Finding.** Substance-related categories (Dangerous Substance + PID) "
            "tripled between 2019 and 2023, consistent with the opioid epidemic's "
            "spillover into Iowa child welfare caseloads.")
    st.markdown("### Figure 4. 2023 Composition")
    show_figure("fig04_2023_composition.png",
                "Figure 4. Distribution of Reported Child Abuse Types in Iowa, 2023.", width=600)
    st.info("**Finding.** Neglect remains dominant at 59.2%, but substance-related "
            "categories together now account for nearly 30% — a structural shift "
            "from the pre-2017 distribution.")
    st.markdown("### Figure 5. Service Area Breakdown")
    show_figure("fig05_service_area.png",
                "Figure 5. Total Reported Occurrences by DHS Service Area, 2010–2023.")
    st.markdown("### Figure 6. Judicial District Breakdown")
    show_figure("fig06_judicial_district.png",
                "Figure 6. Total Reported Child Abuse Occurrences by Iowa Judicial District, 2004–2023.")
    st.markdown("### Figure 7. County-Level Map")
    show_figure("fig07_county_map.png",
                "Figure 7. Geographic Distribution of Total Reported Child Abuse Occurrences by Iowa County, 2004–2023.")
    st.info("**Finding.** Clear urban–rural gradient with visible spatial autocorrelation "
            "between adjacent counties — a hypothesis that motivated the Phase 3 "
            "spatial-feature experiment.")
# ===========================================================================
# PAGE 3 — Phase 1 Count Model
# ===========================================================================
elif page == "Phase 1: Count Model":
    st.title("Phase 1 — XGBoost on Raw Count Outcome")
    st.markdown("""
The first predictive model used the raw count of reported occurrences as the outcome,
with three lagged county counts plus abuse-type composition shares and county fixed
effects as predictors. The result *looked* impressive at first read — but the naive
lag-1 baseline got essentially the same number, which exposed the inflation.
""")
    st.subheader("Rolling-Origin Performance (2018–2023 means)")
    st.dataframe(pd.DataFrame(
        [
            ("Naive (lag1)",  "35.50", "57.20", "0.966"),
            ("Ridge",         "38.01", "58.40", "0.965"),
            ("XGBoost",       "34.21", "54.66", "0.970"),
        ],
        columns=["Model", "MAE (count)", "RMSE (count)", "R² (count)"],
    ), use_container_width=True, hide_index=True)
    st.warning(
        "**The methodological mirage.** Naive lag-1 — predicting each county's "
        "count as its prior year's count, no features at all — gets R² = 0.966. "
        "XGBoost's 0.970 is barely four-thousandths of an R² point above that. "
        "The model isn't learning predictive structure; it's learning that Polk "
        "County is 20x larger than Audubon. This insight drove Phase 2."
    )
    show_figure("fig08_phase1_forecast.png",
                "Figure 8. Phase 1 — Predicted vs. Actual on 2023 Hold-Out (left); "
                "Top 15 Counties by 2024 Forecast Count vs. 2023 Actual (right).")
# ===========================================================================
# PAGE 4 — Phase 2 Rate Model
# ===========================================================================
elif page == "Phase 2: Rate Model":
    st.title("Phase 2 — Per-Capita Rate Model with ACS Denominator")
    st.markdown("""
Phase 2 merged in the U.S. Census Bureau ACS 5-year estimates of population under 18,
reformulated the outcome as occurrences per 1,000 children, and built lagged rate
features. The honest R² this exposes is the project's true predictive performance.
""")
    st.subheader("Phase 2 Rolling-Origin Performance (2018–2023 means)")
    st.dataframe(pd.DataFrame(
        [
            ("Naive (lag1 rate)",  "8.07",  "11.24", "0.141"),
            ("Ridge",              "9.25",  "11.32", "0.093"),
            ("XGBoost",            "7.36",  "9.89",  "0.342"),
        ],
        columns=["Model", "MAE (rate)", "RMSE (rate)", "R² (rate)"],
    ), use_container_width=True, hide_index=True)
    st.success(
        "**The honest baseline.** In rate space, naive lag-1 collapses to R² 0.14. "
        "XGBoost retains 0.342 — real predictive lift over the autoregressive "
        "benchmark. The 2023 count-space MAE of 28.81 (rate predictions × Under18) "
        "actually improves on the Phase 1 count model's 29.34, confirming the "
        "rate model loses no raw-count accuracy."
    )
    st.subheader("Count vs Rate Model — Side by Side")
    st.dataframe(pd.DataFrame(
        [
            ("Naive",   "35.50", "57.20", "0.966",  "8.07", "11.24", "0.141"),
            ("Ridge",   "38.01", "58.40", "0.965",  "9.25", "11.32", "0.093"),
            ("XGBoost", "34.21", "54.66", "0.970",  "7.36",  "9.89", "0.342"),
        ],
        columns=["Model", "Cnt MAE", "Cnt RMSE", "Cnt R²",
                 "Rate MAE", "Rate RMSE", "Rate R²"],
    ), use_container_width=True, hide_index=True)
    show_figure("fig09_phase2_rate.png",
                "Figure 9. Phase 2 — Statewide Rate Trend (left); Top 15 Counties by 2024 Forecast Rate vs. 2023 Actual (right).")
# ===========================================================================
# PAGE 5 — SHAP Interpretability
# ===========================================================================
elif page == "SHAP Interpretability":
    st.title("SHAP — Feature Contributions to the Final Model")
    st.markdown("""
SHAP (SHapley Additive exPlanations) values decompose each prediction into per-feature
contributions. These results are computed on the final 99-county model (no spatial term,
no pre_2014 indicator) over all 990 county-year observations.
""")
    st.subheader("Global Feature Importance")
    st.dataframe(load_shap_importance(),
                 use_container_width=True, hide_index=True)
    st.success(
        "**Key shift from Week 8 → Week 10.** pct_substance overtook pct_physical "
        "for the third importance spot (1.63 vs 1.24), consistent with the post-2017 "
        "substance-related reporting surge visible in Figure 3."
    )
    show_figure("fig10_shap_importance.png",
                "Figure 10. SHAP Global Feature Importance — Final XGBoost Rate Model.")
    show_figure("fig11_shap_beeswarm.png",
                "Figure 11. SHAP Beeswarm — Final XGBoost Rate Model (numeric features).")
# ===========================================================================
# PAGE 6 — Spatial Ablation
# ===========================================================================
elif page == "Spatial Ablation":
    st.title("Phase 3 — Like-for-Like Spatial Ablation")
    st.markdown("""
The Phase 1 EDA's Figure 7 hinted at spatial autocorrelation between adjacent counties.
Week 9 operationalized this with a k=5 nearest-neighbor centroid-based spatial neighbor-lag
feature. Week 10's like-for-like ablation (same 990-row panel, same hyperparameters, only
spatial_lag1 toggled) cleanly answered whether the feature helps.
""")
    st.subheader("Ablation Results")
    st.dataframe(pd.DataFrame(
        [
            ("With spatial_lag1",       "7.542", "10.054", "0.318"),
            ("Without spatial_lag1",    "7.390",  "9.907", "0.339"),
            ("Delta (with − no)",       "+0.152",  "+0.147", "−0.021"),
        ],
        columns=["Configuration", "MAE", "RMSE", "R²"],
    ), use_container_width=True, hide_index=True)
    st.error(
        "**Hypothesis falsified.** The spatial term actively degrades the model "
        "by 0.021 R². It is dropped from the final feature set. A polygon-adjacency "
        "specification (using actual Iowa county boundaries instead of centroid "
        "distance) remains a candidate for future work."
    )
# ===========================================================================
# PAGE 7 — Final 2024 Forecast
# ===========================================================================
elif page == "Final 2024 Forecast":
    st.title("Final 2024 County-Level Forecast")
    fc = load_forecast_data().copy()
    col1, col2, col3 = st.columns(3)
    col1.metric("Statewide 2024 forecast", "16,908")
    col2.metric("Statewide 2023 actual",   "17,188")
    col3.metric("Projected change",        "−1.6%")
    st.markdown("---")
    st.subheader("Top 10 Counties by Forecast Rate (per 1,000)")
    st.markdown("*The prevention-investment ranking — small-population counties "
                "with highest per-child reporting pressure.*")
    top10_rate = fc.nlargest(10, "Forecast_Rate_2024")[
        ["County", "ServiceArea", "Under18_2023",
         "Actual_2023", "Forecast_2024", "Forecast_Rate_2024"]
    ]
    st.dataframe(top10_rate, use_container_width=True, hide_index=True)
    st.subheader("Top 10 Counties by Forecast Count")
    st.markdown("*The resource-allocation ranking — counties where total expected "
                "caseload is highest in 2024.*")
    top10_count = fc.nlargest(10, "Forecast_2024")[
        ["County", "ServiceArea", "Under18_2023",
         "Actual_2023", "Forecast_2024", "Forecast_Rate_2024"]
    ]
    st.dataframe(top10_count, use_container_width=True, hide_index=True)
    show_figure("fig12_2024_forecast.png",
                "Figure 12. Top 15 Iowa Counties — Final XGBoost 2024 Forecast Rate vs. 2023 Actual.")
# ===========================================================================
# PAGE 8 — County Detail Lookup (interactive)
# ===========================================================================
elif page == "County Detail Lookup":
    st.title("County Detail Lookup")
    st.markdown("Pick a county to see its 2024 projection alongside its 2023 baseline.")
    fc = load_forecast_data().copy()
    county = st.selectbox("Select a county",
                          options=sorted(fc["County"].unique()),
                          index=0)
    row = fc[fc["County"] == county].iloc[0]
    st.markdown(f"### {county} County")
    st.markdown(f"*DHS Service Area: {row['ServiceArea']}*")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Under-18 population (2023)", f"{int(row['Under18_2023']):,}")
    c2.metric("2023 Actual Count", f"{int(row['Actual_2023']):,}")
    c3.metric("2024 Forecast Count", f"{int(row['Forecast_2024']):,}",
              delta=f"{int(row['Forecast_2024'] - row['Actual_2023']):+,} vs 2023")
    c4.metric("2024 Forecast Rate (per 1,000)",
              f"{row['Forecast_Rate_2024']:.1f}")
    # County's rate vs statewide rate context
    state_total_2024 = fc["Forecast_2024"].sum()
    state_total_2023 = fc["Actual_2023"].sum()
    state_pop = fc["Under18_2023"].sum()
    state_rate_2024 = state_total_2024 / state_pop * 1000
    st.markdown("---")
    st.subheader("How does this county compare to the rest of Iowa?")
    ratio = row["Forecast_Rate_2024"] / state_rate_2024
    if ratio >= 1.5:
        st.error(
            f"**{county}'s 2024 forecast rate of {row['Forecast_Rate_2024']:.1f} "
            f"per 1,000 is {ratio:.1f}× the partial-sample statewide rate of "
            f"{state_rate_2024:.1f}.** This is a high-intensity county where "
            f"per-child reporting pressure is meaningfully above average."
        )
    elif ratio >= 1.0:
        st.warning(
            f"**{county}'s 2024 forecast rate of {row['Forecast_Rate_2024']:.1f} "
            f"per 1,000 is {ratio:.2f}× the partial-sample statewide rate of "
            f"{state_rate_2024:.1f}** — above the partial-sample state mean but "
            f"not exceptionally so."
        )
    else:
        st.success(
            f"**{county}'s 2024 forecast rate of {row['Forecast_Rate_2024']:.1f} "
            f"per 1,000 is {ratio:.2f}× the partial-sample statewide rate of "
            f"{state_rate_2024:.1f}** — below the partial-sample state mean."
        )
    st.caption(
        "Note: the dashboard ships with the top 25 counties for demo purposes. "
        "The full 99-county forecast is available from the underlying notebook (CB38)."
    )
