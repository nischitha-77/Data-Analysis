import subprocess
import time
import requests
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Backend API URL
BASE_URL = "http://127.0.0.1:8000"

# Function to check if the backend is running
def is_backend_running():
    """Check if backend is running by calling the /ping endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/ping", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Start backend if not running
backend_process = None
if not is_backend_running():
    st.warning("⚠️ Starting backend server... Please wait.")
    
    backend_process = subprocess.Popen(
        ["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    timeout = 10  # Max wait time in seconds
    for _ in range(timeout):
        if is_backend_running():
            st.success("✅ Backend started successfully!")
            break
        time.sleep(1)
    else:
        st.error("❌ Backend failed to start. Please check logs.")

# Streamlit App UI
st.title("🔍 Data Analysis App")
st.subheader("📂 Upload a File for Processing")

# File Upload Section
uploaded_file = st.file_uploader("📤 Choose a file", type=["csv", "xlsx", "json"])

df_preview = None  # Preview DataFrame (first 5 rows)
df_full = None  # Full DataFrame for cleaned data
df_raw = None  # Raw DataFrame for summary & visualization
file_uploaded = False  # Flag to control message display

if uploaded_file is not None:
    file_uploaded = True  # Mark file as uploaded
    file_name = uploaded_file.name

    # Read file contents
    file_bytes = uploaded_file.read()
    files = {"file": (uploaded_file.name, file_bytes)}

    # Send file to backend
    try:
        with st.spinner("⚙️ Processing file..."):
            response = requests.post(f"{BASE_URL}/upload/", files=files)

        if response.status_code == 200:
            data = response.json()

            # Extract preview data (first 5 rows)
            preview_data = data.get("preview", [])
            shape = data.get("shape", {"rows": 0, "columns": 0})

            # Convert preview data to DataFrame
            if preview_data:
                df_preview = pd.DataFrame(preview_data)

            # Fetch full cleaned dataset
            full_data_response = requests.get(f"{BASE_URL}/download/")
            if full_data_response.status_code == 200:
                full_data_bytes = full_data_response.content
                df_full = pd.read_csv(io.BytesIO(full_data_bytes))
            else:
                st.error("❌ Failed to fetch cleaned dataset from backend.")

            # Sidebar options
            st.sidebar.header("🔧 Options")
            option = st.sidebar.radio("Choose an action", [
                "Data Summary",
                "Preprocess Data",
                "Show First N Rows",
                "Visualizations",
                "Download Cleaned Data"
            ])

            # Read raw data locally for summary and visualization
            raw_file_bytes = io.BytesIO(file_bytes)
            file_ext = file_name.split(".")[-1]

            if file_ext == "csv":
                df_raw = pd.read_csv(raw_file_bytes)
            elif file_ext == "xlsx":
                df_raw = pd.read_excel(raw_file_bytes)
            elif file_ext == "json":
                df_raw = pd.read_json(raw_file_bytes)

            # Data Summary Section
            if option == "Data Summary":
                st.write("## 📊 Data Summary")

                if df_raw is not None:
                    # Column Names & Data Types
                    st.write("#### 🏷 Column Names & Data Types")
                    st.dataframe(pd.DataFrame(df_raw.dtypes, columns=["Data Type"]))

                    # Unique Values Per Column
                    st.write("#### 🔢 Unique Values Per Column")
                    st.dataframe(pd.DataFrame(df_raw.nunique(), columns=["Unique Values"]))

                    # Missing Data
                    st.write("#### 📉 Missing Values Per Column")
                    missing_values = pd.DataFrame(df_raw.isnull().sum(), columns=["Missing Values"])
                    st.dataframe(missing_values)

                    # Duplicate Records
                    st.write("#### 📑 Duplicate Records")
                    duplicate_count = df_raw.duplicated().sum()
                    st.write(f"🔁 **Total Duplicates:** {duplicate_count}")

                    # Outliers Detection (Using IQR)
                    st.write("#### 🚨 Outliers Detection (Numerical Columns)")
                    numerical_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
                    if numerical_cols:
                        outlier_counts = {}
                        for col in numerical_cols:
                            Q1 = df_raw[col].quantile(0.25)
                            Q3 = df_raw[col].quantile(0.75)
                            IQR = Q3 - Q1
                            outlier_count = ((df_raw[col] < (Q1 - 1.5 * IQR)) | (df_raw[col] > (Q3 + 1.5 * IQR))).sum()
                            outlier_counts[col] = outlier_count
                        st.dataframe(pd.DataFrame(outlier_counts.items(), columns=["Column", "Outlier Count"]))
                    else:
                        st.write("✅ No numerical columns detected for outlier analysis.")

                    # Inconsistent Formatting
                    st.write("#### 🔄 Inconsistent Formatting")
                    formatting_issues = {}

                    # Check categorical columns for inconsistency
                    categorical_cols = df_raw.select_dtypes(include=['object']).columns.tolist()
                    for col in categorical_cols:
                        unique_values = df_raw[col].dropna().unique()
                        lower_case_unique_values = set([str(val).lower() for val in unique_values])
                        if len(unique_values) != len(lower_case_unique_values):
                            formatting_issues[col] = "⚠️ Case inconsistency detected"

                    # Check date columns
                    for col in df_raw.columns:
                        try:
                            pd.to_datetime(df_raw[col], errors='raise')
                        except Exception:
                            formatting_issues[col] = "⚠️ Possible inconsistent date formats"

                    if formatting_issues:
                        st.dataframe(pd.DataFrame(formatting_issues.items(), columns=["Column", "Issue"]))
                    else:
                        st.write("✅ No formatting issues detected.")

                else:
                    st.error("❌ Unable to load raw data for summary.")

            # Preprocess Data Section
            elif option == "Preprocess Data":
                st.success("✅ File processed successfully!")
                st.write("### 🔍 Preprocessed Data Preview (First 5 Rows)")
                st.dataframe(df_preview)

            # Show First N Rows Section
            elif option == "Show First N Rows":
                num_rows = st.number_input("Enter number of rows to display:", min_value=1, max_value=shape["rows"], value=10, step=1)
                st.write(f"### 🔹 First {num_rows} Rows Preview")
                if df_full is not None:
                    st.dataframe(df_full.head(num_rows))  # Show dynamic N rows
                else:
                    st.warning("⚠️ Full dataset could not be loaded.")

            # Data Visualizations Section
            elif option == "Visualizations":
                st.write("## 📈 Data Visualizations")

                if df_raw is not None and df_full is not None:
                    # Choose dataset (Raw or Cleaned)
                    dataset_choice = st.radio("Select Data for Visualization", ["Raw Data", "Cleaned Data"])

                    if dataset_choice == "Raw Data":
                        df_selected = df_raw
                    else:
                        df_selected = df_full

                    numerical_cols = df_selected.select_dtypes(include=["number"]).columns.tolist()
                    categorical_cols = df_selected.select_dtypes(include=["object"]).columns.tolist()

                    # Box Plot (Outliers)
                    if numerical_cols:
                        st.write("### 📊 Box Plot (Outliers)")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.boxplot(data=df_selected[numerical_cols], ax=ax)
                        st.pyplot(fig)

                    # Bar Chart (Categorical Columns)
                    if categorical_cols:
                        st.write("### 📊 Bar Graph (Categorical Columns)")
                        for col in categorical_cols[:3]:  # Show first 3 categorical columns
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.countplot(x=df_selected[col], ax=ax)
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                    else:
                        st.warning("⚠️ No categorical columns available for bar plots.")

                    # New Visualization 1: Histogram for Numerical Columns
                    if numerical_cols:
                        st.write("### 📊 Histogram (Numerical Columns)")
                        for col in numerical_cols[:3]:  # Show first 3 numerical columns
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.histplot(df_selected[col], kde=True, ax=ax)
                            st.pyplot(fig)

                    # New Visualization 2: Correlation Heatmap
                    if len(numerical_cols) > 1:  # Ensure there are at least two numerical columns
                        st.write("### 📊 Correlation Heatmap")
                        correlation_matrix = df_selected[numerical_cols].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                        st.pyplot(fig)

                else:
                    st.error("❌ No data available for visualization.")

            # Download Cleaned Data Section
            elif option == "Download Cleaned Data":
                st.write("### 📥 Download Cleaned Data")
                if df_full is not None:
                    csv_data = df_full.to_csv(index=False).encode('utf-8')
                    st.download_button(label="⬇️ Download CSV", data=csv_data, file_name="cleaned_data.csv", mime="text/csv")
                else:
                    st.error("❌ No data available for download.")

        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to backend. Make sure it is running.")

st.write("🚀 Ready to analyze your data!")
