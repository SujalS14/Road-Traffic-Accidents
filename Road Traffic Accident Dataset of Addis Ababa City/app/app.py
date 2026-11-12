"""Streamlit interface for exploring injury severity in recorded crashes."""

import joblib
import pandas as pd
import streamlit as st

from train_model import APP_DIR, CATEGORIES, FEATURES, TARGET


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "risk_model.joblib")


@st.cache_data
def load_crashes():
    return pd.read_csv(APP_DIR / "crashes_clean.csv")


def main():
    st.set_page_config(page_title="Road Crash Severity Analysis")
    st.title("Road Crash Severity Analysis")
    st.write("Estimate the chance of serious or fatal injury **given that a crash occurred**.")
    crashes = load_crashes()
    artifact = load_model()

    scorer, coverage, performance = st.tabs(
        ["Score a situation", "Crash data coverage", "Model performance"]
    )
    with scorer:
        with st.form("driving_situation"):
            selected = {}
            for field, label in CATEGORIES.items():
                options = crashes[field].dropna().value_counts().index.tolist()
                selected[field] = st.selectbox(label, options)
            selected["hour"] = st.selectbox(
                "Hour (local time in the dataset)", range(24), index=12,
                format_func=lambda hour: f"{hour:02d}:00",
            )
            submitted = st.form_submit_button("Calculate risk score", type="primary")
        if submitted:
            row = pd.DataFrame([selected], columns=FEATURES)
            probability = artifact["model"].predict_proba(row)[0, 1]
            st.metric("Estimated serious or fatal injury share", f"{probability:.1%}")
            matches = crashes[FEATURES].eq(row.iloc[0]).all(axis=1)
            st.caption(f"{int(matches.sum()):,} recorded crashes match all five selections.")

    with coverage:
        total, severe = st.columns(2)
        total.metric("Recorded crashes", f"{len(crashes):,}")
        severe.metric("Serious or fatal share", f"{crashes[TARGET].mean():.1%}")
        weather = crashes.groupby("Weather_conditions", dropna=False)[TARGET].agg(
            crashes="size", serious_or_fatal_share="mean"
        ).sort_values("crashes")
        st.subheader("Weather conditions with the fewest records")
        st.dataframe(
            weather.reset_index().rename(columns={
                "Weather_conditions": "Weather",
                "crashes": "Recorded crashes",
                "serious_or_fatal_share": "Serious or fatal share",
            }),
            hide_index=True,
            column_config={"Serious or fatal share": st.column_config.NumberColumn(format="percent")},
        )

    with performance:
        metrics = artifact["metrics"]
        accuracy, baseline, auroc = st.columns(3)
        accuracy.metric("Model accuracy", f"{metrics['accuracy']:.1%}")
        baseline.metric("Always slight baseline", f"{metrics['baseline']:.1%}")
        auroc.metric("AUROC", f"{metrics['auroc']:.3f}")
        st.caption(f"Measured on {metrics['test_size']:,} held-out crashes.")
        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual slight", "Actual serious or fatal"],
            columns=["Predicted slight", "Predicted serious or fatal"],
        )
        st.dataframe(matrix)
    st.caption("Source data covers police-reported crashes involving human drivers in Addis Ababa.")


if __name__ == "__main__":
    main()
