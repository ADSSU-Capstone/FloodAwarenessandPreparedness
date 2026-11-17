import io
import re
import string
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Flood Preparedness & Risk Perception Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS: EARTHY & SEA TONES ---
st.markdown(
    """
    <style>
    /* Main Background - Soft Sand */
    .stApp {
        background-color: #FAF9F6;
    }
    
    /* Header - Deep Sea Navy */
    .header-container {
        background: linear-gradient(135deg, #1A3644 0%, #0F2027 100%);
        padding: 30px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-title { 
        font-family: 'Georgia', serif;
        font-size: 2.2rem; 
        font-weight: 700; 
        margin-bottom: 5px;
        color: #F4EADB;
    }
    .main-sub { 
        font-family: 'Arial', sans-serif;
        color: #A3C6C4; 
        font-size: 1rem; 
    }
    
    /* KPI Cards - Seafoam & Sand */
    .kpi-card { 
        background-color: #FFFFFF; 
        border-left: 6px solid #467A7E; 
        border-radius: 4px;
        padding: 20px; 
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .kpi-value { 
        font-family: 'Georgia', serif;
        font-size: 2.2rem; 
        font-weight: 700; 
        color: #1A3644; 
        line-height: 1.1; 
    }
    .kpi-label { 
        font-size: 0.85rem; 
        color: #5A6B6B; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Interpretation Box - Seafoam Tint */
    .interp-box { 
        background-color: #E8F0F0; 
        border-left: 5px solid #467A7E; 
        border-radius: 0 8px 8px 0;
        padding: 15px; 
        font-size: 0.95rem; 
        color: #1A3644; 
        margin: 10px 0 25px 0;
        font-style: italic;
    }
    
    /* Justification Box - Warm Sand */
    .justify-box { 
        background-color: #F4EADB; 
        border-left: 5px solid #C29B6E; 
        border-radius: 0 8px 8px 0;
        padding: 15px; 
        font-size: 0.95rem; 
        color: #4A3A28; 
        margin-bottom: 25px;
    }
    
    /* Warning Box - Terracotta */
    .warn-box { 
        background-color: #FDF0ED; 
        border-left: 5px solid #C06C5C; 
        border-radius: 0 8px 8px 0;
        padding: 15px; 
        font-size: 0.95rem; 
        color: #5C2E25; 
        margin-bottom: 25px;
    }

    /* Custom Tabs - File Folder Style */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        flex-wrap: wrap;
        border-bottom: 2px solid #467A7E;
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: #E8F0F0; 
        border: 1px solid #A3C6C4; 
        border-bottom: none;
        border-radius: 8px 8px 0 0; 
        padding: 10px 18px; 
        font-weight: 600; 
        color: #1A3644;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #1A3644 !important; 
        color: #F4EADB !important; 
        border: 1px solid #1A3644 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F4EADB;
        border-right: 2px solid #C29B6E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-container">
        <div class="main-title">Flood Preparedness & Risk Perception</div>
        <div class="main-sub">A Coastal & Earthy Data Assessment of Community Flood Awareness</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Column Mappings
DEMO_COLS = {
    "AGE": "Age", "SEX": "Sex", "CIVIL STATUS": "Civil Status",
    "EDUCATIONAL ATTAINMENT": "Educational Attainment", "OCCUPATION": "Occupation",
    "MONTHLY HOUSEHOLD INCOME": "Monthly Income", "HOUSEHOLD SIZE (NUMBER OF FAMILY MEMBERS)": "Household Size",
    "BARANGAY": "Barangay"
}

PREP_COLS = [c for c in [
    "FLOOD PREPAREDNESS LEVEL [MY HOUSEHOLD HAS AN EMERGENCY EVACUATION PLAN FOR FLOODING.]",
    "FLOOD PREPAREDNESS LEVEL [WE HAVE PREPARED EMERGENCY SUPPLIES (FOOD, WATER, MEDICINE) FOR FLOOD EVENTS.]",
    "FLOOD PREPAREDNESS LEVEL [I KNOW THE NEAREST EVACUATION CENTER IN OUR AREA.]",
    "FLOOD PREPAREDNESS LEVEL [IMPORTANT DOCUMENTS ARE STORED IN A SAFE AND WATERPROOF PLACE.]",
    "FLOOD PREPAREDNESS LEVEL [WE REGULARLY MONITOR WEATHER UPDATES AND FLOOD WARNINGS.]",
    "FLOOD PREPAREDNESS LEVEL [I KNOW WHAT ACTIONS TO TAKE WHEN FLOOD WARNINGS ARE ISSUED.]",
    "FLOOD PREPAREDNESS LEVEL [MEMBERS OF MY FAMILY KNOW THEIR ASSIGNED ROLES DURING FLOOD EMERGENCIES.]",
    "FLOOD PREPAREDNESS LEVEL [WE HAVE IDENTIFIED SAFE EVACUATION ROUTES IN CASE OF FLOODING.]",
    "FLOOD PREPAREDNESS LEVEL [EMERGENCY TOOLS (FLASHLIGHT, BATTERIES, ETC.) ARE AVAILABLE IN OUR HOME.]",
    "FLOOD PREPAREDNESS LEVEL [WE SECURE IMPORTANT BELONGINGS WHEN FLOODING IS EXPECTED.]",
    "FLOOD PREPAREDNESS LEVEL [I FEEL THAT MY HOUSEHOLD IS PREPARED FOR FLOOD DISASTERS.]"
]]

EXP_COLS = [c for c in [
    "FLOOD EXPERIENCE FACTORS [FLOODWATERS FREQUENTLY REACH OUR RESIDENTIAL AREA.]",
    "FLOOD EXPERIENCE FACTORS [FLOODWATERS ENTER OUR HOUSE.]",
    "FLOOD EXPERIENCE FACTORS [OUR PROPERTY HAS BEEN DAMAGED DUE TO FLOODING.]",
    "FLOOD EXPERIENCE FACTORS [FLOODING DISRUPTS OUR DAILY ACTIVITIES.]",
    "FLOOD EXPERIENCE FACTORS [FLOODING AFFECTS TRANSPORTATION IN OUR AREA.]",
    "FLOOD EXPERIENCE FACTORS [OUR HOUSEHOLD HAS BEEN EVACUATED DUE TO FLOODING.]",
    "FLOOD EXPERIENCE FACTORS [FLOODING CAUSES POWER INTERRUPTIONS IN OUR AREA.]",
    "FLOOD EXPERIENCE FACTORS [FLOODING LIMITS ACCESS TO CLEAN WATER.]",
    "FLOOD EXPERIENCE FACTORS [FLOODING AFFECTS OUR LIVELIHOOD OR SOURCE OF INCOME.]",
    "FLOOD EXPERIENCE FACTORS [ROADS IN OUR AREA BECOME IMPASSABLE DURING FLOOD EVENTS.]",
    "FLOOD EXPERIENCE FACTORS [WE EXPERIENCE FINANCIAL LOSSES DUE TO FLOODING.]",
    "FLOOD EXPERIENCE FACTORS [FLOODING AFFECTS SCHOOL OR WORK ATTENDANCE.]",
    "FLOOD EXPERIENCE FACTORS [OUR COMMUNITY EXPERIENCES RECURRING FLOOD EVENTS.]"
]]

INFO_COLS = [c for c in [
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [I RECEIVE FLOOD WARNINGS IN A TIMELY MANNER WHENEVER FLOODING IS EXPECTED.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [I KNOW WHERE TO OBTAIN OFFICIAL FLOOD INFORMATION.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [THE BARANGAY REGULARLY PROVIDES FLOOD PREPAREDNESS UPDATES.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [I UNDERSTAND THE FLOOD WARNING SIGNALS USED IN OUR COMMUNITY.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [SOCIAL MEDIA HELPS ME STAY INFORMED ABOUT FLOOD RISKS.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [COMMUNITY MEMBERS COOPERATE EFFECTIVELY DURING FLOOD EMERGENCIES.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [RELIEF GOODS ARE DISTRIBUTED PROMPTLY AFTER FLOOD EVENTS.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [THE LOCAL GOVERNMENT RESPONDS PROMPTLY DURING FLOOD EMERGENCIES.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [VOLUNTEERS PROVIDE TIMELY ASSISTANCE DURING FLOOD EMERGENCIES.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [COORDINATION BETWEEN COMMUNITY AND GOVERNMENT IS EFFECTIVE DURING FLOODS.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [RESIDENTS ARE ENCOURAGED TO PARTICIPATE IN FLOOD PREPAREDNESS ACTIVITIES.]",
    "ACCESS TO DISASTER INFORMATION AND COMMUNITY SUPPORT [DISASTER RESPONSE IN OUR COMMUNITY IS WELL-ORGANIZED.]"
]]

RISK_COLS = [c for c in [
    "RISK PERCEPTION [MY AREA IS HIGHLY VULNERABLE TO FLOODING.]",
    "RISK PERCEPTION [FLOODING POSES A SERIOUS RISK TO MY FAMILY.]",
    "RISK PERCEPTION [FLOOD PREPAREDNESS CAN REDUCE FLOOD DAMAGE.]",
    "RISK PERCEPTION [I AM CONCERNED ABOUT FUTURE FLOODING EVENTS.]",
    "RISK PERCEPTION [CLIMATE CHANGE MAY INCREASE FLOOD RISKS IN OUR AREA.]",
    "RISK PERCEPTION [FLOODING CAN SERIOUSLY AFFECT HEALTH AND SAFETY.]",
    "RISK PERCEPTION [LACK OF PREPARATION INCREASES FLOOD IMPACTS.]",
    "RISK PERCEPTION [PREPARING BEFORE FLOODS IS BETTER THAN REACTING DURING EMERGENCIES.]",
    "RISK PERCEPTION [EVERY HOUSEHOLD SHOULD PREPARE FOR FLOODING.]",
    "RISK PERCEPTION [FLOOD RISKS IN OUR COMMUNITY ARE INCREASING OVER TIME.]",
    "RISK PERCEPTION [I FEEL RESPONSIBLE FOR PREPARING FOR FLOOD DISASTERS.]",
    "RISK PERCEPTION [DISASTER PREPAREDNESS IS IMPORTANT EVEN WHEN THERE IS NO FLOODING.]"
]]

TEXT_COL_CHALLENGE = "IN YOUR OPINION, WHAT IS THE BIGGEST CHALLENGE THAT PREVENTS YOUR HOUSEHOLD FROM BEING FULLY PREPARED FOR FLOODS?"
TEXT_COL_SUGGESTION = "WHAT PROGRAMS, ASSISTANCE, OR SUPPORT FROM THE LOCAL GOVERNMENT UNIT (LGU) OR MDRRMO WOULD HELP IMPROVE YOUR HOUSEHOLD'S FLOOD PREPAREDNESS?"

def strip_invisible(text):
    if not isinstance(text, str): return text
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text):
    if not isinstance(text, str): return ""
    t = strip_invisible(text).lower()
    return t.translate(str.maketrans("", "", string.punctuation))

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [strip_invisible(str(c)).strip().upper() for c in df.columns]
    return df

def prep_data(df):
    df = df.copy()
    
    # Map Demographic columns cleanly
    for raw_col, clean_col in DEMO_COLS.items():
        match = [c for c in df.columns if raw_col in c]
        if match:
            df[clean_col] = df[match[0]].apply(strip_invisible)
            
    # Calculate Composites (1-5 scale)
    for c in PREP_COLS + EXP_COLS + INFO_COLS + RISK_COLS:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df["Preparedness_Score"] = df[PREP_COLS].mean(axis=1)
    df["Experience_Score"] = df[EXP_COLS].mean(axis=1)
    df["Info_Score"] = df[INFO_COLS].mean(axis=1)
    df["Risk_Score"] = df[RISK_COLS].mean(axis=1)
    
    # Binarize Preparedness for Modeling (Median split: High vs Low)
    median_prep = df["Preparedness_Score"].median()
    df["High_Preparedness"] = (df["Preparedness_Score"] >= median_prep).astype(int)
    
    # Text processing
    if TEXT_COL_CHALLENGE in df.columns:
        df["Challenge_Text"] = df[TEXT_COL_CHALLENGE].apply(strip_invisible)
        df["Challenge_Clean"] = df["Challenge_Text"].apply(clean_text)
    else:
        df["Challenge_Text"] = ""
        df["Challenge_Clean"] = ""
        
    return df

def interp(text):
    st.markdown(f'<div class="interp-box">{text}</div>', unsafe_allow_html=True)

def justify(text):
    st.markdown(f'<div class="justify-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def kpi(value, label):
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("### Dataset Upload")
file_up = st.sidebar.file_uploader("Upload Flood Awareness Dataset", type=["csv", "xlsx", "xls"])

if file_up is None:
    st.info("Please upload the Flood Awareness dataset to load the dashboard.")
    st.stop()

try:
    raw_df = load_data(file_up.getvalue(), file_up.name)
    df = prep_data(raw_df)
except Exception as e:
    st.error(f"Error reading the file: {e}")
    st.stop()

if len(df) == 0:
    st.info("No valid records found in the uploaded dataset.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Drill-Down Filters")
barangay_list = ["All"] + sorted(df["Barangay"].dropna().unique().tolist()) if "Barangay" in df.columns else ["All"]
focus = st.sidebar.selectbox("Barangay Location", barangay_list)
filtered = df if focus == "All" else df[df["Barangay"] == focus]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Valid Records:** {len(df)}")
st.sidebar.markdown(f"**Currently Viewing:** {len(filtered)}")

# --- TABS ---
tabs = st.tabs([
    "Data Overview & Profiling",
    "Preparedness vs Experience",
    "Access to Information",
    "Risk Perception",
    "Patterns & Clustering",
    "Predictive Model Evaluation",
    "Qualitative Challenges"
])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1: kpi(len(filtered), "Valid Households")
    with c2: kpi(f"{filtered['Preparedness_Score'].mean():.2f} / 5", "Mean Preparedness Score")
    with c3: kpi(f"{filtered['Risk_Score'].mean():.2f} / 5", "Mean Risk Perception")
    
    justify(
        "Data cleaning involved mapping raw survey columns to their respective thematic blocks (Preparedness, Experience, Info Access, Risk Perception). "
        "Missing Likert responses were coerced to NaN and composites were calculated via arithmetic mean. The Preparedness score was binarized "
        "at the median to serve as the target variable for the predictive models."
    )
    
    if "Barangay" in filtered.columns:
        res = filtered["Barangay"].value_counts().reset_index()
        res.columns = ["Barangay", "Respondents"]
        bar = px.bar(res.sort_values("Respondents"), x="Respondents", y="Barangay", orientation="h",
                     title="Respondent Distribution by Barangay", color="Respondents", color_continuous_scale="Teal")
        bar.update_layout(height=400, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, use_container_width=True)
        interp(f"Within the current selection, {res.iloc[0,0]} has the highest number of respondents. "
               "This distribution highlights the density of survey coverage across target areas.")

    st.markdown("**Demographic Profile**")
    demo_cols = [c for c in ["Age", "Sex", "Civil Status", "Educational Attainment", "Occupation", "Monthly Income"] if c in filtered.columns]
    if demo_cols:
        chosen = st.selectbox("Select Demographic Variable", demo_cols)
        vc = filtered[chosen].value_counts().reset_index()
        vc.columns = [chosen, "Count"]
        bar2 = px.bar(vc, x=chosen, y="Count", title=f"Distribution of {chosen}", color="Count", color_continuous_scale="Teal")
        bar2.update_layout(height=400, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar2, use_container_width=True)
        interp(f"The modal category for {chosen} is {vc.iloc[0,0]}. Understanding this demographic baseline ensures that programmatic recommendations "
               "are targeted to the dominant population segment.")

with tabs[1]:
    if "Preparedness_Score" in filtered.columns and "Experience_Score" in filtered.columns:
        sc = px.scatter(
            filtered, x="Experience_Score", y="Preparedness_Score",
            color=filtered["High_Preparedness"].map({1: "High Preparedness", 0: "Low Preparedness"}),
            title="Flood Experience vs. Preparedness Level",
            labels={"Experience_Score": "Experience Severity (1-5)", "Preparedness_Score": "Preparedness Level (1-5)"},
            opacity=0.7,
            color_discrete_map={"High Preparedness": "#467A7E", "Low Preparedness": "#C06C5C"}
        )
        sc.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(sc, use_container_width=True)
        interp("The scatter plot reveals the relationship between how severely households experience flooding and how prepared they are. "
               "Observations in the upper-right indicate households that face high flood frequency but have adapted with high preparedness. "
               "Those in the lower-right face high flood severity but are critically under-prepared.")
        
        box = px.box(
            filtered, x="High_Preparedness", y="Experience_Score",
            title="Flood Experience Severity by Preparedness Group",
            labels={"High_Preparedness": "Preparedness Class (0=Low, 1=High)", "Experience_Score": "Experience Severity"},
            color="High_Preparedness",
            color_discrete_sequence=["#C06C5C", "#467A7E"]
        )
        box.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(box, use_container_width=True)
        interp("By comparing the median experience scores across the two preparedness groups, we can assess whether severe historical flooding "
               "serves as a catalyst for households to become more prepared.")
    justify("Flood preparedness and experience are treated as composite variables. Pearson correlation is calculated to determine if there is a linear "
            "relationship between experiencing severe floods and taking proactive measures.")

with tabs[2]:
    if "Info_Score" in filtered.columns:
        kpi(f"{filtered['Info_Score'].mean():.2f} / 5", "Mean Access to Information Score")
        
        hist = px.histogram(filtered, x="Info_Score", nbins=10, title="Distribution of Access to Disaster Information",
                            color_discrete_sequence=["#1A3644"])
        hist.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(hist, use_container_width=True)
        interp(f"The average access to information score is {filtered['Info_Score'].mean():.2f}. A right-skewed distribution indicates that most "
               "households feel they receive timely warnings and know where to find official information.")
        
        if "Barangay" in filtered.columns:
            area_info = filtered.groupby("Barangay")["Info_Score"].mean().reset_index()
            area_info = area_info.sort_values("Info_Score", ascending=False)
            bar = px.bar(area_info, x="Info_Score", y="Barangay", orientation="h",
                         title="Average Access to Information by Barangay",
                         color="Info_Score", color_continuous_scale="Teal")
            bar.update_layout(height=400, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, use_container_width=True)
            best_area = area_info.iloc[0]
            worst_area = area_info.iloc[-1]
            interp(f"{best_area['Barangay']} has the highest average access to information ({best_area['Info_Score']:.2f}), "
                   f"while {worst_area['Barangay']} has the lowest ({worst_area['Info_Score']:.2f}). "
                   "This disparity highlights where communication channels need strengthening.")
    justify("Access to information is measured as the mean of the 12 items in the Info block. Grouping by barangay "
            "allows local government units to pinpoint areas where warning dissemination is failing.")

with tabs[3]:
    if "Risk_Score" in filtered.columns:
        kpi(f"{filtered['Risk_Score'].mean():.2f} / 5", "Mean Risk Perception Score")
        
        if "Preparedness_Score" in filtered.columns:
            sc = px.scatter(
                filtered, x="Risk_Score", y="Preparedness_Score",
                title="Risk Perception vs. Actual Preparedness",
                labels={"Risk_Score": "Risk Perception (1-5)", "Preparedness_Score": "Preparedness (1-5)"},
                color="Preparedness_Score", color_continuous_scale="Teal", opacity=0.7
            )
            sc.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(sc, use_container_width=True)
            interp("This chart highlights the gap between perception and action. A cluster in the upper-left indicates households that "
                   "recognize the high risk but fail to prepare (the 'knowledge-action gap'). A cluster in the lower-right indicates households "
                   "that are highly prepared but do not perceive the risk as severe.")
            
            corr_val = filtered["Risk_Score"].corr(filtered["Preparedness_Score"])
            justify(f"Pearson correlation between Risk Perception and Preparedness is {corr_val:.2f}. A weak or negative correlation "
                    "suggests that simply raising risk awareness is insufficient; households may lack the financial or logistical means to act on that risk.")

with tabs[4]:
    corr_cols = [c for c in ["Preparedness_Score", "Experience_Score", "Info_Score", "Risk_Score"] if c in filtered.columns]
    if len(corr_cols) >= 2:
        corr = filtered[corr_cols].corr(method="pearson")
        heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Teal", zmin=-1, zmax=1,
                         title="Pearson Correlation Matrix of Composite Scores")
        heat.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(heat, use_container_width=True)
        interp("The correlation matrix reveals which factors move together. A strong positive correlation between Info_Score and Preparedness_Score "
               "suggests that access to timely information is a key driver of preparedness.")
    
    if len(corr_cols) >= 2 and len(filtered) >= 5:
        X = filtered[corr_cols].dropna()
        if len(X) >= 5:
            k = st.slider("Number of clusters (K-Means)", 2, 5, 3)
            Xs = StandardScaler().fit_transform(X)
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
            cdf = X.copy()
            cdf["Cluster"] = labels
            fig = px.scatter(cdf, x="Experience_Score", y="Preparedness_Score", color=cdf["Cluster"].astype(str),
                             title=f"K-Means Clustering of Households (K={k})", opacity=0.8,
                             color_discrete_sequence=px.colors.sequential.Teal)
            fig.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cdf.groupby("Cluster")[corr_cols].mean().round(2), use_container_width=True)
            interp("Clustering segments households into distinct risk profiles. For example, one cluster may represent 'High Risk / Low Preparedness' "
                   "households that require immediate intervention, while another may represent 'High Info / High Preparedness' households.")
        justify("K-Means is applied to standardized composite scores to group households with similar characteristics, experiences, and perceptions. "
                "Standardization is necessary because the composite scores, while nominally identical, carry different variance profiles.")

with tabs[5]:
    if "High_Preparedness" in filtered.columns and len(filtered) >= 20:
        X = filtered[["Experience_Score", "Info_Score", "Risk_Score"]].dropna()
        y = filtered.loc[X.index, "High_Preparedness"]
        
        if X.shape[1] > 0 and len(y.unique()) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }
            results = []
            for name, mdl in models.items():
                mdl.fit(X_train, y_train)
                pred = mdl.predict(X_test)
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "F1-Score": f1_score(y_test, pred, average="macro"),
                })
            res_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            bar = px.bar(res_df, x="Model", y="F1-Score", title="Model Performance Comparison (Target: High vs. Low Preparedness)",
                         color="Model", color_discrete_sequence=["#1A3644", "#467A7E", "#C29B6E"])
            bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, use_container_width=True)
            best = res_df.iloc[0]
            interp(f"The best-performing model is {best['Model']} with an F1-Score of {best['F1-Score']:.3f}. "
                   "This indicates that flood experience, access to information, and risk perception are meaningful predictors "
                   "of whether a household achieves high preparedness.")
        else:
            warn("Insufficient features or target classes to train a predictive model.")
    else:
        warn("Dataset is too small or missing one class to perform stratified predictive modeling.")
    justify("The classification task predicts binary preparedness (high vs. low based on the median split). "
            "Logistic Regression, Random Forest, and Gradient Boosting are compared to determine which algorithm best captures "
            "the relationship between prior experience, information access, and actual preparedness.")

with tabs[6]:
    st.markdown("**Most Common Challenges Preventing Preparedness**")
    if "Challenge_Clean" in filtered.columns:
        all_words = " ".join(filtered["Challenge_Clean"].dropna().tolist()).split()
        stopwords = {"ang", "ng", "sa", "na", "ug", "kay", "the", "a", "is", "of", "and", "to", "in", "we", "our", "for", "my", "it", "on", "by", "with", "that", "this"}
        words = [w for w in all_words if len(w) > 3 and w not in stopwords]
        freq = Counter(words).most_common(15)
        
        if freq:
            freq_df = pd.DataFrame(freq, columns=["Keyword", "Frequency"])
            bar = px.bar(freq_df, x="Frequency", y="Keyword", orientation="h", title="Top Keywords in 'Biggest Challenge' Responses",
                         color="Frequency", color_continuous_scale="Teal")
            bar.update_layout(height=500, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, use_container_width=True)
            interp("The most frequent keywords in the open-ended challenge responses reveal the root causes preventing preparedness. "
                   "Terms like 'kulang', 'kwarta', 'financial', and 'information' highlight that both economic constraints and communication gaps "
                   "are the primary barriers.")
        else:
            warn("No text data available for keyword extraction.")
            
    st.markdown("**Sample Recommendations for LGU / MDRRMO**")
    if TEXT_COL_SUGGESTION in filtered.columns:
        samples = filtered[TEXT_COL_SUGGESTION].dropna().head(10)
        st.dataframe(samples, use_container_width=True)
        interp("The qualitative suggestions provided by residents offer direct, actionable policy recommendations. "
               "These responses frequently point to the need for early warning systems, financial assistance, and physical infrastructure "
               "improvements such as evacuation centers and drainage.")
    justify("Qualitative text analysis is essential to complement the quantitative Likert scores. It captures the nuance of household vulnerabilities "
            "that numerical scales might miss.")
