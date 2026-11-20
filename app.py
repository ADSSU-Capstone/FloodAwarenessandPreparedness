import io
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Flood Preparedness Classification — Bunawan",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {background:#F7F8FA;}
.hero {background:linear-gradient(120deg,#123B52,#1D6070);padding:24px 28px;
       border-radius:12px;color:white;margin-bottom:18px;}
.hero h1 {color:white;font-size:1.8rem;margin:0 0 6px 0;}
.hero p {color:#E4F0F2;margin:0;}
.metric-card {background:white;border:1px solid #E5E9EF;border-left:5px solid #287D8E;
              border-radius:8px;padding:14px 16px;min-height:90px;}
.metric-value {font-size:1.8rem;font-weight:700;color:#123B52;}
.metric-label {font-size:.78rem;color:#596B75;text-transform:uppercase;letter-spacing:.04em;}
.note {background:#EAF3F5;border-left:4px solid #287D8E;padding:12px 14px;
       border-radius:5px;margin:10px 0;}
.warning {background:#FFF3E7;border-left:4px solid #D18A38;padding:12px 14px;
          border-radius:5px;margin:10px 0;}
</style>
<div class="hero">
  <h1>Classification of Flood Preparedness Levels in Bunawan</h1>
  <p>Survey-Based Machine Learning Techniques | Agusan del Sur, Philippines</p>
</div>
""", unsafe_allow_html=True)

CLASS_ORDER = ["Low", "Moderate", "High"]
TARGET_LOW, TARGET_MID = 2.33, 3.66

@st.cache_data(show_spinner=False)
def read_dataset(file_bytes, filename):
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        data = pd.read_csv(bio)
    else:
        data = pd.read_excel(bio)
    data.columns = [str(c).strip().upper() for c in data.columns]
    return data

def find_part_columns(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]

def identify_columns(df):
    # The uploaded cleaned workbook follows the questionnaire's 58-column
    # order: Part I (8), Part II (11), Part III (13), Part IV (12),
    # Part V (12), Part VI (2).
    part1 = df.columns[0:8].tolist()
    part2 = df.columns[8:19].tolist()
    part3 = df.columns[19:32].tolist()
    part4 = df.columns[32:44].tolist()
    part5 = df.columns[44:56].tolist()
    part6 = df.columns[56:58].tolist()

    if len(df.columns) != 58:
        raise ValueError(
            f"Expected the 58 questionnaire columns in the cleaned dataset, "
            f"but found {len(df.columns)}. Please upload the cleaned workbook "
            f"with Part I–VI columns in the original questionnaire order."
        )
    return {
        "part1": part1, "part2": part2, "part3": part3,
        "part4": part4, "part5": part5, "part6": part6,
        "predictors": part1 + part3 + part4 + part5,
    }

def metric_card(value, label):
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True
    )

def add_note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)

def add_warning(text):
    st.markdown(f'<div class="warning">{text}</div>', unsafe_allow_html=True)

def score_preparedness(data, part2_cols):
    part2 = data[part2_cols].apply(pd.to_numeric, errors="coerce")
    # Chapter 3 requires all 11 items to be valid; no partial mean.
    complete = part2.notna().all(axis=1) & part2.ge(1).all(axis=1) & part2.le(5).all(axis=1)
    mean_score = part2.mean(axis=1).round(2).where(complete)
    category = pd.Series(index=data.index, dtype="object")
    category.loc[mean_score.between(1.00, 2.33, inclusive="both")] = "Low"
    category.loc[mean_score.between(2.34, 3.66, inclusive="both")] = "Moderate"
    category.loc[mean_score.between(3.67, 5.00, inclusive="both")] = "High"
    return mean_score, category

def ordinal_orders(columns):
    orders = {}
    for c in columns:
        name = c.upper()
        if name == "AGE":
            orders[c] = ["18–27", "28–37", "38–47", "48–57", "58–67"]
        elif name == "EDUCATIONAL ATTAINMENT":
            orders[c] = [
                "No Formal Education", "Elementary", "Junior High School",
                "Senior High School", "College", "Postgraduate"
            ]
        elif name == "MONTHLY HOUSEHOLD INCOME":
            orders[c] = [
                "Below ₱10,000", "₱10,000–₱20,000",
                "₱20,001–₱40,000", "Above ₱40,000"
            ]
        elif name.startswith("HOUSEHOLD SIZE"):
            orders[c] = ["1–3", "4–6", "7–9", "10 or more"]
    return orders

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    orders = ordinal_orders(categorical_cols)
    ordinal_cols = [c for c in categorical_cols if c in orders]
    nominal_cols = [c for c in categorical_cols if c not in ordinal_cols]

    transformers = []
    if ordinal_cols:
        ord_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[orders[c] for c in ordinal_cols],
                handle_unknown="use_encoded_value",
                unknown_value=np.nan
            )),
            # CategoricalNB accepts only non-negative category codes.
            # Replace any unseen-category NaN with the most frequent code.
            ("encoded_imputer", SimpleImputer(strategy="most_frequent"))
        ])
        transformers.append(("ordinal", ord_pipe, ordinal_cols))
    if nominal_cols:
        nominal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("nominal", nominal_pipe, nominal_cols))
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0
    )

def make_models(X):
    return {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced",
            n_jobs=-1
        ),
        "Categorical Naive Bayes": CategoricalNB(alpha=1.0, min_categories=20)
    }

def make_pipeline(preprocessor, classifier):
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

def clean_open_text(series):
    return series.fillna("").astype(str).str.strip().replace({"nan": ""})

def build_theme_summary(series, theme_terms):
    texts = clean_open_text(series).str.lower()
    rows = []
    for theme, terms in theme_terms.items():
        count = int(texts.apply(lambda t: any(term in t for term in terms)).sum())
        rows.append({"Theme": theme, "Responses mentioning theme": count})
    return pd.DataFrame(rows).sort_values("Responses mentioning theme", ascending=False)

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload cleaned flood awareness dataset",
    type=["xlsx", "xls", "csv"]
)
if uploaded is None:
    st.info("Upload **Flood awareness dataset (data clean).xlsx** to view the analysis.")
    st.stop()

try:
    raw = read_dataset(uploaded.getvalue(), uploaded.name)
    colsets = identify_columns(raw)
except Exception as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.stop()

# Exact duplicate rows are removed; source workbook has 410 records.
before = len(raw)
data = raw.drop_duplicates().copy()
duplicate_count = before - len(data)

data["Flood_Preparedness_Score"], data["Flood_Preparedness_Level"] = score_preparedness(
    data, colsets["part2"]
)
usable = data[data["Flood_Preparedness_Level"].notna()].copy()

st.sidebar.write(f"Records in uploaded file: **{before:,}**")
st.sidebar.write(f"Exact duplicate rows removed: **{duplicate_count:,}**")
st.sidebar.write(f"Records with valid Part II target: **{len(usable):,}**")

# Filter display/summary tabs by barangay; model uses full valid dataset by default.
barangay_col = next((c for c in colsets["part1"] if c == "BARANGAY"), None)
if barangay_col:
    barangays = ["All"] + sorted(usable[barangay_col].dropna().astype(str).unique().tolist())
    chosen_barangay = st.sidebar.selectbox("Barangay filter (descriptive tabs)", barangays)
    display_df = usable if chosen_barangay == "All" else usable[usable[barangay_col].astype(str) == chosen_barangay]
else:
    chosen_barangay = "All"
    display_df = usable.copy()

tabs = st.tabs([
    "Overview & Respondent Profile",
    "Preparedness Classification",
    "Model Development & Evaluation",
    "Predictor Importance",
    "Flood Experience & Risk Factors",
    "Challenges & Suggested Interventions",
    "Data Quality"
])

with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{len(usable):,}", "Valid households")
    with c2: metric_card(f"{len(colsets['predictors'])}", "Predictor variables")
    with c3: metric_card("3", "Preparedness categories")
    with c4: metric_card(f"{len(colsets['part6'])}", "Open-ended questions")

    add_note(
        "The machine-learning target is derived only from the 11 Part II items. "
        "Predictors come from Parts I, III, IV, and V. Part II items are excluded "
        "from model inputs to prevent target leakage; Part VI is analyzed separately."
    )
    st.markdown("#### Respondent Profile")
    profile_vars = [c for c in colsets["part1"] if c in display_df.columns]
    profile_choice = st.selectbox("Choose profile variable", profile_vars)
    profile = display_df[profile_choice].fillna("Not specified").astype(str).value_counts().reset_index()
    profile.columns = [profile_choice, "Households"]
    fig = px.bar(profile, x=profile_choice, y="Households",
                 title=f"Households by {profile_choice}")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(profile, hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Household Flood Preparedness Classification")
    counts = usable["Flood_Preparedness_Level"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    pct = (counts / counts.sum() * 100).round(2)
    c1, c2, c3 = st.columns(3)
    for col, cls in zip([c1, c2, c3], CLASS_ORDER):
        with col:
            metric_card(f"{counts[cls]:,} ({pct[cls]:.1f}%)", f"{cls} preparedness")
    plot_df = pd.DataFrame({
        "Preparedness Level": CLASS_ORDER,
        "Households": [counts[x] for x in CLASS_ORDER],
        "Percentage": [pct[x] for x in CLASS_ORDER]
    })
    fig = px.bar(plot_df, x="Preparedness Level", y="Households",
                 color="Preparedness Level", text="Percentage",
                 title="Distribution of Flood Preparedness Levels")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Preparedness Score Summary")
    score_summary = usable["Flood_Preparedness_Score"].describe().to_frame("Score").round(3)
    st.dataframe(score_summary, use_container_width=True)
    st.markdown("#### Classification Rules")
    st.markdown(
        "- **Low:** 1.00–2.33\n"
        "- **Moderate:** 2.34–3.66\n"
        "- **High:** 3.67–5.00"
    )
    add_note(
        "Each household score is the mean of all 11 Part II items, rounded to two "
        "decimal places. A household is classified only when all 11 items have "
        "valid scores from 1 to 5. The 11 Part II items are positively worded, "
        "so no reverse coding is applied."
    )
    st.markdown("#### Household-level classification records")
    st.dataframe(
        usable[[barangay_col, "Flood_Preparedness_Score", "Flood_Preparedness_Level"]]
        if barangay_col else usable[["Flood_Preparedness_Score", "Flood_Preparedness_Level"]],
        use_container_width=True
    )

@st.cache_resource(show_spinner=True)
def train_and_evaluate(data_for_model, predictor_cols, target_col):
    X = data_for_model[predictor_cols].copy()
    y = data_for_model[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    min_class_train = int(y_train.value_counts().min())
    n_splits = 5
    if min_class_train < n_splits:
        raise ValueError(
            f"Five-fold stratified cross-validation requires at least 5 training "
            f"records in every class. The smallest training class has {min_class_train}. "
            "Review the class distribution or collect more observations."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scorer = make_scorer(
        f1_score, labels=CLASS_ORDER, average="macro", zero_division=0, pos_label=None
    )
    model_defs = make_models(X)
    cv_rows = []
    fitted_candidates = {}

    for name, classifier in model_defs.items():
        prep = build_preprocessor(X)
        pipe = make_pipeline(prep, classifier)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={
                "macro_f1": scorer,
                "accuracy": "accuracy",
                "weighted_f1": "f1_weighted"
            },
            n_jobs=1,
            return_train_score=False,
            error_score="raise"
        )
        cv_rows.append({
            "Algorithm": name,
            "Mean CV Macro-F1": scores["test_macro_f1"].mean(),
            "SD CV Macro-F1": scores["test_macro_f1"].std(),
            "Mean CV Accuracy": scores["test_accuracy"].mean(),
            "Mean CV Weighted-F1": scores["test_weighted_f1"].mean()
        })
        fitted_candidates[name] = (pipe, scores)

    cv_results = pd.DataFrame(cv_rows).sort_values(
        "Mean CV Macro-F1", ascending=False
    ).reset_index(drop=True)
    best_name = cv_results.iloc[0]["Algorithm"]
    best_pipe = fitted_candidates[best_name][0]
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    final_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro-Precision": precision_score(
            y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "Macro-Recall": recall_score(
            y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "Macro-F1": f1_score(
            y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "Weighted-F1": f1_score(
            y_test, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0
        )
    }
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    report = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, zero_division=0, output_dict=True
    )
    test_rows = X_test.copy()
    test_rows["Actual"] = y_test
    test_rows["Predicted"] = y_pred

    # Permutation importance shuffles original columns, grouping the influence
    # of encoded categories under their original questionnaire variable.
    perm = permutation_importance(
        best_pipe, X_test, y_test, scoring=scorer,
        n_repeats=20, random_state=42, n_jobs=1
    )
    importance = pd.DataFrame({
        "Predictor": predictor_cols,
        "Mean Importance": perm.importances_mean,
        "SD": perm.importances_std
    }).sort_values("Mean Importance", ascending=False)

    fold_rows = []
    for model_name, (_, model_scores) in fitted_candidates.items():
        for fold_index in range(len(model_scores["test_macro_f1"])):
            fold_rows.append({
                "Algorithm": model_name,
                "Fold": f"Fold {fold_index + 1}",
                "Macro-F1": model_scores["test_macro_f1"][fold_index],
                "Accuracy": model_scores["test_accuracy"][fold_index],
                "Weighted-F1": model_scores["test_weighted_f1"][fold_index]
            })

    return {
        "cv_results": cv_results,
        "cv_folds": pd.DataFrame(fold_rows),
        "best_name": best_name,
        "metrics": final_metrics, "confusion": cm, "report": report,
        "importance": importance, "predictions": test_rows,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "model": best_pipe
    }

with tabs[2]:
    st.subheader("Decision Tree, Random Forest, and Naïve Bayes")
    st.caption("Model comparison uses stratified 5-fold cross-validation on the 80% training set. The 20% test set is reserved for final evaluation of the selected model.")
    class_counts = usable["Flood_Preparedness_Level"].value_counts()
    if len(usable) < 25 or len(class_counts) < 3 or class_counts.min() < 10:
        add_warning(
            "The preparedness classes are imbalanced, especially the Low category. "
            "The current dataset contains very few Low-preparedness households. "
            "Although stratified splitting is used, class-specific scores for Low "
            "may be unstable and should be interpreted cautiously."
        )
    try:
        results = train_and_evaluate(
            usable, colsets["predictors"], "Flood_Preparedness_Level"
        )
        st.markdown("#### The three classification algorithms in this study")
        st.caption(
            "All three algorithms specified in Chapter 3 are trained and compared "
            "using the same predictor variables, training split, and five-fold "
            "cross-validation partitions."
        )

        algorithm_descriptions = {
            "Decision Tree": (
                "A tree-based classifier. It splits households using predictor "
                "conditions and assigns Low, Moderate, or High preparedness at "
                "the terminal leaves. The implementation uses the Gini Impurity criterion."
            ),
            "Random Forest": (
                "An ensemble of decision trees. Each tree produces a class prediction, "
                "and the final household category is determined by majority voting."
            ),
            "Categorical Naive Bayes": (
                "A probabilistic classifier based on Bayes' Theorem and the conditional "
                "independence assumption. Categorical Naïve Bayes is used as the initial "
                "variant, with predictor values encoded as non-negative category codes."
            )
        }
        algorithm_order = [
            "Decision Tree", "Random Forest", "Categorical Naive Bayes"
        ]
        algorithm_columns = st.columns(3)
        for algorithm_name, algorithm_col in zip(algorithm_order, algorithm_columns):
            row = results["cv_results"].loc[
                results["cv_results"]["Algorithm"] == algorithm_name
            ].iloc[0]
            with algorithm_col:
                st.markdown("**" + algorithm_name + "**")
                st.markdown(algorithm_descriptions[algorithm_name])
                metric_card(
                    f"{row['Mean CV Macro-F1']:.3f}",
                    "Mean 5-fold CV Macro-F1"
                )
                st.caption(
                    f"CV Accuracy: {row['Mean CV Accuracy']:.3f} · "
                    f"CV Weighted-F1: {row['Mean CV Weighted-F1']:.3f}"
                )

        st.markdown("#### Cross-validation comparison")
        st.dataframe(
            results["cv_results"].round(4),
            hide_index=True,
            use_container_width=True
        )
        cv_plot = px.bar(
            results["cv_results"].sort_values("Mean CV Macro-F1"),
            x="Mean CV Macro-F1", y="Algorithm", orientation="h",
            error_x="SD CV Macro-F1",
            title="Five-fold Cross-Validation: Mean Macro-F1"
        )
        st.plotly_chart(cv_plot, use_container_width=True)

        with st.expander("View the five-fold scores for each algorithm"):
            st.dataframe(
                results["cv_folds"].pivot(
                    index="Fold", columns="Algorithm", values="Macro-F1"
                ).round(4),
                use_container_width=True
            )
            st.caption(
                "Each value is the validation Macro-F1 for that fold. "
                "The algorithm with the highest mean Macro-F1 is selected."
            )

        st.success(
            f"Selected best-performing model: {results['best_name']} — "
            "highest mean five-fold cross-validation Macro-F1."
        )

        st.markdown("#### Training and test class distribution")
        split_counts = pd.DataFrame({
            "Class": CLASS_ORDER,
            "Training records": [int((results["y_train"] == c).sum()) for c in CLASS_ORDER],
            "Testing records": [int((results["y_test"] == c).sum()) for c in CLASS_ORDER]
        })
        st.dataframe(split_counts, hide_index=True, use_container_width=True)
        if (split_counts["Testing records"] < 5).any():
            add_warning(
                "At least one class has fewer than five records in the reserved test set. "
                "This makes its final class-specific metrics highly uncertain. "
                "The current dataset contains only a small number of Low-preparedness "
                "households, so the model may fail to identify that category reliably."
            )

        st.markdown("#### Final evaluation on the reserved 20% test set")
        metric_cols = st.columns(5)
        for col, (label, value) in zip(metric_cols, results["metrics"].items()):
            with col:
                metric_card(f"{value:.3f}", label)

        cm_fig = px.imshow(
            results["confusion"], x=CLASS_ORDER, y=CLASS_ORDER,
            text_auto=True, aspect="auto",
            labels={"x": "Predicted Class", "y": "Actual Class", "color": "Count"},
            title=f"Confusion Matrix — {results['best_name']}"
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        report_df = pd.DataFrame(results["report"]).T
        st.markdown("#### Classification report")
        st.dataframe(report_df.round(3), use_container_width=True)
        with st.expander("View held-out test predictions"):
            st.dataframe(results["predictions"], use_container_width=True)
        add_note(
            "The algorithm is selected using mean Macro-F1 from 5-fold "
            "cross-validation on training data only. The reserved test set is "
            "used once for the final metrics of that selected algorithm."
        )
    except Exception as exc:
        results = None
        st.error(f"Model training/evaluation could not be completed: {exc}")

with tabs[3]:
    st.subheader("Most Influential Predictor Variables")
    try:
        results
    except NameError:
        results = None
    if results is None:
        st.info("Complete the model evaluation first. If training failed, review the class counts and dataset.")
    else:
        imp = results["importance"].head(15).sort_values("Mean Importance")
        fig = px.bar(
            imp, x="Mean Importance", y="Predictor", orientation="h",
            error_x="SD", title="Top Predictor Variables — Permutation Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results["importance"].round(5), hide_index=True, use_container_width=True)
        add_note(
            "Permutation importance measures the decrease in the selected model's "
            "Macro-F1 when one original predictor column is shuffled. It is an "
            "association with model performance, not evidence that the variable "
            "directly causes household preparedness."
        )

with tabs[4]:
    st.subheader("Flood Experience, Information Access, Community Support, and Risk Perception")
    construct_map = {
        "Flood Experience Factors (Part III)": colsets["part3"],
        "Access to Disaster Information and Community Support (Part IV)": colsets["part4"],
        "Risk Perception (Part V)": colsets["part5"],
    }
    construct_means = []
    for label, cols in construct_map.items():
        vals = usable[cols].apply(pd.to_numeric, errors="coerce")
        construct_means.append({
            "Construct": label,
            "Mean Score": vals.stack().mean()
        })
    construct_df = pd.DataFrame(construct_means)
    fig = px.bar(construct_df, x="Construct", y="Mean Score",
                 title="Mean Scores by Predictor Construct")
    fig.update_yaxes(range=[1, 5])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Item-level descriptive means")
    selected_construct = st.selectbox("Select construct", list(construct_map.keys()))
    cols = construct_map[selected_construct]
    item_means = usable[cols].apply(pd.to_numeric, errors="coerce").mean().sort_values().reset_index()
    item_means.columns = ["Survey Item", "Mean"]
    item_fig = px.bar(item_means, x="Mean", y="Survey Item", orientation="h",
                      title=f"Item Means — {selected_construct}")
    item_fig.update_xaxes(range=[1, 5])
    st.plotly_chart(item_fig, use_container_width=True)
    st.dataframe(item_means.round(3), hide_index=True, use_container_width=True)

    st.markdown("#### Preparedness level by barangay")
    if barangay_col:
        cross = pd.crosstab(usable[barangay_col], usable["Flood_Preparedness_Level"])
        cross = cross.reindex(columns=CLASS_ORDER, fill_value=0)
        st.dataframe(cross, use_container_width=True)
        fig = px.bar(cross.reset_index(), x=barangay_col, y=CLASS_ORDER,
                     title="Preparedness Classification by Barangay",
                     barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Household Challenges and Suggested Interventions")
    challenge_col, support_col = colsets["part6"]
    challenges = clean_open_text(usable[challenge_col])
    suggestions = clean_open_text(usable[support_col])
    c1, c2 = st.columns(2)
    with c1:
        metric_card(f"{(challenges != '').sum():,}", "Challenge responses")
    with c2:
        metric_card(f"{(suggestions != '').sum():,}", "Suggested intervention responses")

    st.markdown("#### Common challenge terms")
    stopwords = {
        "the","and","for","that","this","with","from","have","has","had","are","was",
        "were","our","their","they","them","there","about","because","would","could",
        "should","can","not","but","also","very","more","much","when","what","how",
        "why","which","who","all","any","just","into","during","household","flood",
        "floods","flooding","prepared","preparation","prepare","preparing","need",
        "needs","help","also","mga","ang","nga","sa","kay","unya","wala","dili"
    }
    def term_counts(series):
        words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", " ".join(series.tolist()).lower())
        return Counter(w for w in words if w not in stopwords)
    challenge_terms = term_counts(challenges)
    suggestion_terms = term_counts(suggestions)

    if challenge_terms:
        term_df = pd.DataFrame(challenge_terms.most_common(20), columns=["Term", "Frequency"])
        fig = px.bar(term_df.sort_values("Frequency"), x="Frequency", y="Term",
                     orientation="h", title="Most Frequent Terms in Challenge Responses")
        st.plotly_chart(fig, use_container_width=True)
    if suggestion_terms:
        term_df2 = pd.DataFrame(suggestion_terms.most_common(20), columns=["Term", "Frequency"])
        fig = px.bar(term_df2.sort_values("Frequency"), x="Frequency", y="Term",
                     orientation="h", title="Most Frequent Terms in Suggested Interventions")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Read and review actual responses")
    which = st.radio("Response type", ["Household challenges", "Suggested LGU/MDRRMO support"], horizontal=True)
    if which == "Household challenges":
        response_df = usable[[challenge_col]].rename(columns={challenge_col: "Response"})
    else:
        response_df = usable[[support_col]].rename(columns={support_col: "Response"})
    response_df = response_df[response_df["Response"].notna() & (response_df["Response"].astype(str).str.strip() != "")]
    st.dataframe(response_df, use_container_width=True)
    add_note(
        "This tab provides keyword frequencies and the original open-ended answers "
        "for qualitative review. These summaries are exploratory aids; they do not "
        "replace the study's thematic analysis and manual interpretation of responses."
    )

with tabs[6]:
    st.subheader("Data Quality and Preparation Checks")
    qc = pd.DataFrame({
        "Check": [
            "Uploaded records",
            "Exact duplicate rows removed",
            "Records retained after duplicate removal",
            "Records with complete and valid 11-item Part II target",
            "Records excluded from classification due to invalid/missing Part II",
            "Predictor cells missing (before imputation)",
            "Missing challenge responses",
            "Missing suggested-intervention responses"
        ],
        "Result": [
            before, duplicate_count, len(data), len(usable),
            len(data) - len(usable),
            int(usable[colsets["predictors"]].isna().sum().sum()),
            int(usable[challenge_col].isna().sum()),
            int(usable[support_col].isna().sum())
        ]
    })
    st.dataframe(qc, hide_index=True, use_container_width=True)
    st.markdown("#### Missing predictor values by variable")
    missing = usable[colsets["predictors"]].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        miss_df = missing.reset_index()
        miss_df.columns = ["Predictor", "Missing Values"]
        st.dataframe(miss_df, hide_index=True, use_container_width=True)
    else:
        st.write("No missing values were found in predictor variables in the usable dataset.")
    add_note(
        "Missing predictor values are imputed within each model pipeline: categorical "
        "variables use the most frequent category. The preprocessing pipeline is fitted "
        "inside each cross-validation fold and then refitted on the full training set "
        "before evaluation on the held-out test set."
    )

st.caption(
    "Methodology source: Chapter 3 of “Classification of Flood Preparedness Levels "
    "in Bunawan Through Survey-Based Machine Learning Techniques.” "
    "This dashboard is for analysis and visualization; it is not a real-time "
    "flood forecasting or disaster-management system."
)
