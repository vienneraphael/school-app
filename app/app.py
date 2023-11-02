import os

from utils import compute_manual_score

# Ensure we are one the good directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Student failure dashboard", page_icon="🧑‍🎓", layout="wide")


def main():
    st.title('Student support dashboard')

    st.markdown(
        """
        This dashboard aims to provide a decision tool for teachers which want to have insights about the improvement capacity of their students.
        Two interactives visualisations are implemented:
        - The first one uses an improvability score estimated using machine learning.
        - The second one uses an handcrafted improvability score which enables the user to parameter it by selecting the individual weights of three differents factors.
        """
    )

    # =============================================================================
    # Load data
    # =============================================================================
    st.header("Data")

    st.markdown("You can drop a csv file for custom data. Else, the default dataset is already loaded.")
    # Base dataset path
    base_path = "data/student_data.csv"

    # Enable the user to upload its own dataset
    csv_data = st.file_uploader("Select your own csv file:", type=['csv'])
    data_path = csv_data if csv_data is not None else base_path
    data = pd.read_csv(data_path)

    data["FullName"] = data.FirstName + " " + data.FamilyName

    # =============================================================================
    # School selection
    # =============================================================================

    st.header("School selection")

    all_options_school = st.checkbox("Select all schools (Uncheck if you want to select a subset of schools):", True)

    selected_school = st.multiselect(
        'Select schools to include in the analysis:', data.school.unique(), disabled=all_options_school
    )
    if all_options_school:
        selected_school = data.school.unique()
    # =============================================================================
    # Students notes selection
    # =============================================================================

    st.header("Students filtering")
    student_filter = st.slider(
        "Select a threshold for filtering out students having grades superior than this threshold.",
        max_value=20,
        min_value=0,
        value=20,
        step=1,
    )
    idx_mask = data.FinalGrade.le(student_filter) & data.school.isin(selected_school)

    # =============================================================================
    # ML-based improvability score
    # =============================================================================
    st.header("ML-based improvability score")
    # clean_df = pd.get_dummies(data)
    model = joblib.load("models/booster.joblib")

    # potential_grade = model.predict(clean_df[FEATURES])
    improvability_score_ml = model.predict(data)

    data["improvability_score_ml"] = improvability_score_ml
    fig = px.box(
        data[idx_mask],
        x='FinalGrade',
        y="improvability_score_ml",
        hover_data=["FullName"],
        labels={"improvability_score_ml": "Improvability Score (ML)"},
        points="all",
    )
    fig.update_xaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """You will find more information about each student by moving the mouse over
                the data points."""
    )

    # =============================================================================
    # Manual improvability score
    # =============================================================================

    st.header("Manual improvability score")
    st.markdown(
        """
    Move coefficients according to what criteria are the most important for a student to be **improvable**.
    Select the coefficients based on your conception of which factor impact the most the improvability of a student
    """
    )

    # Select the parameters used to compute the improvability score
    keys = ["studytime", "Dalc", "absences"]
    names = ["Study Time", "Daytime alcohol consumption", "Absences"]
    col1, col2 = st.columns([2, 1])
    col2.markdown("***")

    # Create a slider for each parameter
    coeff_dict = {}
    for k, name in zip(keys, names):
        coeff_dict.update({k: col2.slider(name, 0, 10, 5)})

    # Compute the manual improvability score
    score = compute_manual_score(data, coeff_dict)

    data["improvability_score_manual"] = score

    st.markdown(
        """
                The graph above is updated on real time based on your selection.
                You will find more information about each student by moving the mouse over the data points.
                """
    )

    fig = px.box(
        data[idx_mask],
        x="FinalGrade",
        y="improvability_score_manual",
        hover_data=["FullName"] + keys,
        labels={"improvability_score_manual": "Improvability Score (Manual)"},
        points="all",
    )
    fig.update_xaxes(autorange="reversed")

    col1.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
