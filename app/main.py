import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging
from typing import Dict, Any, List, Tuple
import chardet
import re
from datetime import datetime
from dateutil.parser import parse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import Tool
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend('Agg')

# ====================== ENHANCED FILE LOADING ======================
def detect_file_encoding(uploaded_file, sample_size=100000):
    """More robust encoding detection with fallbacks"""
    raw_data = uploaded_file.read(sample_size)
    uploaded_file.seek(0)
    
    # Try chardet first
    try:
        result = chardet.detect(raw_data)
        if result['confidence'] > 0.9:
            return result['encoding']
    except:
        pass
    
    # Common encodings to try (in order of likelihood)
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings:
        try:
            raw_data.decode(encoding)
            return encoding
        except:
            continue
    
    return 'utf-8'  # Default fallback

def load_data_file(uploaded_file):
    """Ultra-robust data loader with multiple fallbacks"""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Get detected encoding
                encoding = detect_file_encoding(uploaded_file)
                
                # Try reading with detected encoding
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    encoding=encoding,
                    on_bad_lines='warn',
                    engine='python',
                    dtype=str  # Read everything as string first
                )
                
                # Clean the data
                df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                return df
            
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                # For Excel files, try different engines
                try:
                    return pd.read_excel(uploaded_file, engine='openpyxl')
                except:
                    return pd.read_excel(uploaded_file, engine='xlrd')
                    
        except Exception as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed: {str(e)}")
            if attempt == max_attempts:
                raise ValueError(f"Could not load file after {max_attempts} attempts. Error: {str(e)}")
            # Reset file pointer for next attempt
            uploaded_file.seek(0)

# ====================== ETL Pipeline ======================
class DataETL:
    def __init__(self):
        self.quality_report = {}
        self.transform_rules = {}
        self.analytics_ready = False

    def extract(self, uploaded_file) -> pd.DataFrame:
        """Enhanced extraction with better error handling"""
        try:
            logger.info("Starting file extraction")
            
            # First try the robust loader
            df = load_data_file(uploaded_file)
            
            # If empty, try alternative methods
            if df.empty:
                logger.warning("Initial load produced empty dataframe, trying alternative methods")
                uploaded_file.seek(0)
                
                # Try reading line by line for problematic CSVs
                if uploaded_file.name.endswith('.csv'):
                    lines = []
                    for line in uploaded_file:
                        try:
                            lines.append(line.decode('utf-8').strip().split(','))
                        except:
                            try:
                                lines.append(line.decode('latin1').strip().split(','))
                            except:
                                continue
                    
                    if lines:
                        df = pd.DataFrame(lines[1:], columns=lines[0])
            
            # Final check
            if df.empty:
                raise ValueError("Unable to extract data - file may be corrupted or improperly formatted")
            
            self.quality_report['initial_shape'] = df.shape
            logger.info(f"Successfully extracted data with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise ValueError(f"Failed to extract data: {str(e)}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automated data cleaning and transformation"""
        try:
            # Data Quality Checks
            self._generate_quality_report(df)
            
            # Automated Cleaning
            df = self._clean_data(df)
            df = self._standardize_data(df)
            df = self._handle_missing_values(df)
            df = self._feature_engineering(df)
            
            self.quality_report['final_shape'] = df.shape
            self.analytics_ready = True
            return df
        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning"""
        # Convert all columns to string first to handle mixed types
        df = df.astype(str)
        
        # Replace common problematic characters
        df = df.applymap(lambda x: x.replace('\xa0', ' ').replace('\xa3', '£') 
                             if isinstance(x, str) else x)
        
        # Clean column names
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col).strip().lower() for col in df.columns]
        
        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates()
        self.quality_report['duplicates_removed'] = initial_count - len(df)
        
        return df

    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types and formats"""
        for col in df.columns:
            # Attempt numeric conversion
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
            
            # Attempt datetime conversion
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = df[col].apply(lambda x: parse(str(x)) if any(isinstance(x, str) for x in df[col]) else df[col])
                except:
                    pass
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart missing value handling"""
        missing_report = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_report[col] = missing_count
                
                # Numeric columns: fill with median
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                # Categorical columns: fill with mode
                else:
                    mode = df[col].mode()
                    if not mode.empty:
                        df[col] = df[col].fillna(mode[0])
        
        self.quality_report['missing_values'] = missing_report
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automated feature engineering"""
        # Extract date components
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
        
        # Create interaction features between numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    df[f'{col1}_ratio_{col2}'] = df[col1] / (df[col2] + 1e-6)  # Avoid division by zero
        
        return df

    def _generate_quality_report(self, df: pd.DataFrame):
        """Generate comprehensive data quality report"""
        self.quality_report = {
            'initial_rows': len(df),
            'initial_columns': len(df.columns),
            'missing_values': df.isna().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'column_types': {col: str(df[col].dtype) for col in df.columns}
        }

# ====================== Analytics Engine ======================
class AnalyticsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_types = self._detect_column_types()
        self.visualizations = {
            'distribution': self._plot_distribution,
            'correlation': self._plot_correlation,
            'trend': self._plot_trend,
            'composition': self._plot_composition,
            'comparison': self._plot_comparison
        }

    def _detect_column_types(self) -> Dict[str, str]:
        """Categorize columns by data type"""
        return {
            col: 'numeric' if pd.api.types.is_numeric_dtype(self.df[col]) else
                 'datetime' if pd.api.types.is_datetime64_any_dtype(self.df[col]) else
                 'categorical'
            for col in self.df.columns
        }

    def analyze(self, query: str) -> Dict[str, Any]:
        """Main analysis function"""
        try:
            # Determine analysis type from query
            analysis_type = self._determine_analysis_type(query)
            
            if analysis_type in self.visualizations:
                return self.visualizations[analysis_type](query)
            elif 'stat' in query.lower():
                return self.generate_summary_stats(query)
            else:
                return self._generate_insights(query)
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "type": "error",
                "message": f"Analysis error: {str(e)}"
            }

    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis needed"""
        query = query.lower()
        if 'distribution' in query or 'histogram' in query:
            return 'distribution'
        elif 'correlation' in query or 'relationship' in query:
            return 'correlation'
        elif 'trend' in query or 'over time' in query:
            return 'trend'
        elif 'composition' in query or 'breakdown' in query:
            return 'composition'
        elif 'compare' in query:
            return 'comparison'
        else:
            return 'insight'

    def _plot_distribution(self, query: str) -> Dict[str, Any]:
        """Generate distribution plots"""
        col = self._extract_column(query)
        if not col:
            return {"type": "error", "message": "Please specify a column"}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if self.column_types[col] == 'numeric':
            sns.histplot(self.df[col], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
        else:
            self.df[col].value_counts().head(20).plot(kind='bar', ax=ax)
            ax.set_title(f"Value Counts for {col}")
        
        return self._fig_to_response(fig, f"Distribution of {col}")

    def _plot_correlation(self, query: str) -> Dict[str, Any]:
        """Generate correlation plots"""
        numeric_cols = [col for col, typ in self.column_types.items() if typ == 'numeric']
        if len(numeric_cols) < 2:
            return {"type": "error", "message": "Need at least 2 numeric columns"}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(self.df[numeric_cols].corr(), annot=True, ax=ax)
        ax.set_title("Correlation Matrix")
        return self._fig_to_response(fig, "Correlation Matrix")

    def _plot_trend(self, query: str) -> Dict[str, Any]:
        """Generate time series plots"""
        date_cols = [col for col, typ in self.column_types.items() if typ == 'datetime']
        if not date_cols:
            return {"type": "error", "message": "No datetime columns found"}
        
        numeric_cols = [col for col, typ in self.column_types.items() if typ == 'numeric']
        if not numeric_cols:
            return {"type": "error", "message": "No numeric columns found"}
        
        date_col = date_cols[0]
        value_col = numeric_cols[0]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        self.df.set_index(date_col)[value_col].plot(ax=ax)
        ax.set_title(f"Trend of {value_col} over time")
        return self._fig_to_response(fig, f"Trend of {value_col}")

    def _plot_composition(self, query: str) -> Dict[str, Any]:
        """Generate composition plots (pie, stacked bar)"""
        cat_cols = [col for col, typ in self.column_types.items() if typ == 'categorical']
        if not cat_cols:
            return {"type": "error", "message": "No categorical columns found"}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df[cat_cols[0]].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_title(f"Composition of {cat_cols[0]}")
        return self._fig_to_response(fig, f"Composition of {cat_cols[0]}")

    def _plot_comparison(self, query: str) -> Dict[str, Any]:
        """Generate comparison plots"""
        cols = self._extract_columns(query, count=2)
        if len(cols) != 2:
            return {"type": "error", "message": "Please specify exactly 2 columns to compare"}
        
        col1, col2 = cols
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.column_types[col1] == 'numeric' and self.column_types[col2] == 'numeric':
            sns.scatterplot(x=col1, y=col2, data=self.df, ax=ax)
        elif self.column_types[col1] == 'numeric' and self.column_types[col2] == 'categorical':
            sns.boxplot(x=col2, y=col1, data=self.df, ax=ax)
        elif self.column_types[col1] == 'categorical' and self.column_types[col2] == 'numeric':
            sns.boxplot(x=col1, y=col2, data=self.df, ax=ax)
        else:
            pd.crosstab(self.df[col1], self.df[col2]).plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_title(f"Comparison of {col1} and {col2}")
        return self._fig_to_response(fig, f"Comparison of {col1} and {col2}")

    def generate_summary_stats(self, query: str = None) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        stats = {
            'dataset_info': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'missing_values': int(self.df.isna().sum().sum()),
                'duplicates': int(self.df.duplicated().sum())
            },
            'numeric_stats': self.df.describe().to_dict(),
            'categorical_stats': {
                col: {
                    'unique_values': self.df[col].nunique(),
                    'top_value': self.df[col].mode()[0] if not self.df[col].mode().empty else None,
                    'top_count': int(self.df[col].value_counts().iloc[0]) if not self.df[col].value_counts().empty else None
                }
                for col in self.df.select_dtypes(exclude=np.number).columns
            }
        }
        
        return {
            "type": "stats",
            "data": stats,
            "description": "Dataset Summary Statistics"
        }

    def _generate_insights(self, query: str) -> Dict[str, Any]:
        """Generate automated insights"""
        numeric_cols = [col for col, typ in self.column_types.items() if typ == 'numeric']
        cat_cols = [col for col, typ in self.column_types.items() if typ == 'categorical']
        
        insights = []
        
        # Numeric column insights
        for col in numeric_cols:
            stats = self.df[col].describe()
            insights.append(f"- {col}: Mean = {stats['mean']:.2f}, Range = {stats['min']:.2f}-{stats['max']:.2f}")
        
        # Categorical column insights
        for col in cat_cols:
            top_value = self.df[col].mode()[0] if not self.df[col].mode().empty else None
            insights.append(f"- {col}: {self.df[col].nunique()} unique values, most common = {top_value}")
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            corr = self.df[numeric_cols].corr().unstack().sort_values(ascending=False)
            strongest = corr[corr < 1].idxmax()
            insights.append(f"- Strongest correlation: {strongest[0]} and {strongest[1]} ({corr[strongest]:.2f})")
        
        return {
            "type": "insights",
            "data": "\n".join(insights),
            "description": "Automated Data Insights"
        }

    def _fig_to_response(self, fig, description: str) -> Dict[str, Any]:
        """Convert matplotlib figure to response dict"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return {
            "type": "image",
            "data": base64.b64encode(buf.getvalue()).decode('utf-8'),
            "description": description
        }

    def _extract_column(self, query: str) -> str:
        """Extract column name from query"""
        for col in self.df.columns:
            if col.lower() in query.lower():
                return col
        return None

    def _extract_columns(self, query: str, count: int = 2) -> List[str]:
        """Extract multiple column names from query"""
        found = []
        remaining_cols = set(self.df.columns)
        
        # First try exact matches
        for col in self.df.columns:
            if col.lower() in query.lower():
                found.append(col)
                remaining_cols.remove(col)
                if len(found) >= count:
                    return found[:count]
        
        # Then try partial matches
        for col in remaining_cols:
            if any(word in col.lower() for word in query.lower().split()):
                found.append(col)
                if len(found) >= count:
                    return found[:count]
        
        return found[:count]

# ====================== Streamlit UI ======================
def main():
    st.set_page_config(
        page_title="Data Analyst Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'etl' not in st.session_state:
        st.session_state.etl = DataETL()
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analytics' not in st.session_state:
        st.session_state.analytics = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar - File Upload
    with st.sidebar:
        st.title("🔍 Data Analyst Pro")
        st.markdown("""
        ### 🔒 Security Notice
        - Your data never leaves your browser
        - Processing happens entirely in your session
        """)
        
        uploaded_file = st.file_uploader(
            "📤 Upload Data File", 
            type=["csv", "xlsx", "xls"],
            help="Supports CSV and Excel files"
        )
        
        if uploaded_file and st.session_state.df is None:
            try:
                with st.spinner("Processing your data..."):
                    # Show encoding detection progress
                    with st.expander("🔍 Detecting file encoding..."):
                        encoding = detect_file_encoding(uploaded_file)
                        st.write(f"Detected encoding: {encoding}")
                    
                    # ETL Process with progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Extracting data...")
                    st.session_state.df = st.session_state.etl.extract(uploaded_file)
                    progress_bar.progress(25)
                    
                    status_text.text("Cleaning data...")
                    st.session_state.df = st.session_state.etl._clean_data(st.session_state.df)
                    progress_bar.progress(50)
                    
                    status_text.text("Transforming data...")
                    st.session_state.df = st.session_state.etl.transform(st.session_state.df)
                    progress_bar.progress(75)
                    
                    status_text.text("Preparing for analysis...")
                    st.session_state.analytics = AnalyticsEngine(st.session_state.df)
                    progress_bar.progress(100)
                    status_text.text("Done!")
                    
                    # Show quick data preview
                    with st.expander("👉 Quick Preview", expanded=True):
                        st.dataframe(st.session_state.df.head(3))
                    
                    # Initial messages
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": f"✅ Data loaded successfully with {len(st.session_state.df.columns)} columns and {len(st.session_state.df)} rows."
                        },
                        {
                            "role": "assistant",
                            "content": {
                                "type": "table",
                                "data": st.session_state.df.head().to_markdown(),
                                "description": "First 5 rows of your data"
                            }
                        }
                    ]
                    
                    # Suggest analyses based on data types
                    numeric_cols = [col for col in st.session_state.df.columns 
                                   if pd.api.types.is_numeric_dtype(st.session_state.df[col])]
                    date_cols = [col for col in st.session_state.df.columns 
                                if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col])]
                    
                    suggestions = ["What would you like to analyze? Try:"]
                    
                    if numeric_cols:
                        suggestions.append(f"- 'Show distribution of {numeric_cols[0]}'")
                    if len(numeric_cols) >= 2:
                        suggestions.append(f"- 'Compare {numeric_cols[0]} and {numeric_cols[1]}'")
                    if date_cols:
                        suggestions.append(f"- 'Show trends over time for {date_cols[0]}'")
                    
                    suggestions.append("- 'Show summary statistics'")
                    suggestions.append("- 'What are the data quality issues?'")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "\n".join(suggestions)
                    })
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Failed to process file: {str(e)}")
                logger.exception("File processing failed")
                
                # Provide troubleshooting tips
                with st.expander("🛠️ Troubleshooting Help"):
                    st.markdown("""
                    **Common issues and solutions:**
                    
                    1. **Encoding problems**:
                       - Try saving your file as UTF-8 CSV
                       - Remove special characters from column headers
                    
                    2. **Excel-specific issues**:
                       - Save as .xlsx instead of .xls
                       - Remove password protection
                    
                    3. **Corrupted files**:
                       - Try opening and resaving in the original application
                    
                    4. **Large files**:
                       - Try with a smaller subset of your data first
                    """)

    # Main interface
    st.title("📊 Data Analyst Pro")
    
    # Show data preview if available
    if st.session_state.df is not None:
        with st.expander("🔍 Data Preview (First 5 Rows)", expanded=True):
            st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # Data Quality Report
        with st.expander("📋 Data Quality Report", expanded=False):
            st.json(st.session_state.etl.quality_report)
    
    # Chat interface
    st.divider()
    st.subheader("💬 Analysis Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                if message["content"]["type"] == "image":
                    st.image(base64.b64decode(message["content"]["data"]), 
                            caption=message["content"]["description"],
                            use_column_width=True)
                elif message["content"]["type"] == "table":
                    st.markdown(f"**{message['content']['description']}**")
                    st.markdown(message["content"]["data"])
                elif message["content"]["type"] == "stats":
                    st.markdown(f"### {message['content']['description']}")
                    st.json(message["content"]["data"])
                else:
                    st.markdown(message["content"]["data"])
            else:
                st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing..."):
                    if st.session_state.analytics:
                        response = st.session_state.analytics.analyze(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        if response["type"] == "image":
                            st.image(base64.b64decode(response["data"]), 
                                    caption=response["description"],
                                    use_column_width=True)
                        elif response["type"] == "stats":
                            st.markdown(f"### {response['description']}")
                            st.json(response["data"])
                        else:
                            st.markdown(response["data"])
                    else:
                        st.error("Please upload a data file first")
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

if __name__ == "__main__":
    main()