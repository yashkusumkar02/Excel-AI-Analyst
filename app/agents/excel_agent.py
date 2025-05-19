import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging
from typing import Dict, Any, List, Union
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_experimental")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.switch_backend('Agg')  # Required for non-interactive environments

class SmartExcelAnalyst:
    def __init__(self, df: pd.DataFrame, api_key: str = None):
        self.df = self._clean_data(df.copy())
        self.api_key = api_key
        if api_key:
            self.llm = self._initialize_llm()
            self.memory = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",
                return_messages=True
            )
        self.column_types = self._detect_column_types()
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically clean and preprocess data"""
        logger.info("Cleaning and preprocessing data")
        
        # Convert all columns to string first to handle mixed types
        for col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', 'N/A', ''], np.nan)
        
        # Attempt type conversion
        for col in df.columns:
            # Try numeric conversion first
            try:
                df[col] = pd.to_numeric(df[col])
                continue
            except (ValueError, TypeError):
                pass
            
            # Try datetime conversion
            try:
                df[col] = pd.to_datetime(df[col])
                continue
            except (ValueError, TypeError):
                pass
            
            # Keep as string if other conversions fail
            df[col] = df[col].astype(str)
        
        return df

    def _detect_column_types(self) -> Dict[str, str]:
        """Categorize columns by their data type"""
        column_categories = {}
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                column_categories[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                column_categories[col] = 'datetime'
            else:
                column_categories[col] = 'categorical'
        
        return column_categories

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize Gemini LLM if API key is provided"""
        if not self.api_key:
            return None
            
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.3,
            max_output_tokens=4000,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    def _create_visualization(self, plot_func, title, xlabel="", ylabel="") -> Dict[str, Any]:
        """Helper function to create and save visualizations"""
        try:
            plt.figure(figsize=(10, 6))
            plot_func()
            plt.title(title)
            if xlabel: plt.xlabel(xlabel)
            if ylabel: plt.ylabel(ylabel)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            return {
                "type": "image",
                "data": base64.b64encode(buf.getvalue()).decode('utf-8'),
                "description": title
            }
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return {
                "type": "error",
                "message": f"Could not generate visualization: {str(e)}"
            }

    def auto_visualize(self, query: str = None) -> Dict[str, Any]:
        """Automatically generate appropriate visualizations based on data"""
        numeric_cols = [col for col, typ in self.column_types.items() if typ == 'numeric']
        categorical_cols = [col for col, typ in self.column_types.items() if typ == 'categorical']
        datetime_cols = [col for col, typ in self.column_types.items() if typ == 'datetime']
        
        # If specific query provided, try to match it
        if query:
            query = query.lower()
            
            # Check for distribution requests
            if 'distribution' in query or 'histogram' in query:
                for col in numeric_cols:
                    if col.lower() in query:
                        return self._create_visualization(
                            lambda: sns.histplot(self.df[col], bins=30, kde=True),
                            f"Distribution of {col}",
                            col,
                            "Frequency"
                        )
            
            # Check for relationship requests
            if 'relationship' in query or 'correlation' in query and len(numeric_cols) >= 2:
                return self._create_visualization(
                    lambda: sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=self.df),
                    f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                    numeric_cols[0],
                    numeric_cols[1]
                )
            
            # Check for top N requests
            if 'top' in query and categorical_cols and numeric_cols:
                top_n = 10
                if 'top' in query:
                    try:
                        top_n = int(query.split('top')[1].strip().split()[0])
                    except:
                        pass
                
                return self._create_visualization(
                    lambda: (
                        self.df.groupby(categorical_cols[0])[numeric_cols[0]]
                        .sum().nlargest(top_n)
                        .plot(kind='barh')
                    ),
                    f"Top {top_n} {categorical_cols[0]} by {numeric_cols[0]}"
                )
        
        # Default visualizations when no specific query or no match found
        if numeric_cols:
            if len(numeric_cols) >= 2:
                # Show relationship between first two numeric columns
                return self._create_visualization(
                    lambda: sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=self.df),
                    f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                    numeric_cols[0],
                    numeric_cols[1]
                )
            else:
                # Show distribution of first numeric column
                return self._create_visualization(
                    lambda: sns.histplot(self.df[numeric_cols[0]], bins=30, kde=True),
                    f"Distribution of {numeric_cols[0]}",
                    numeric_cols[0],
                    "Frequency"
                )
        elif categorical_cols:
            # Show value counts for first categorical column
            return self._create_visualization(
                lambda: self.df[categorical_cols[0]].value_counts().plot(kind='bar'),
                f"Distribution of {categorical_cols[0]}"
            )
        else:
            return {
                "type": "error",
                "message": "No suitable columns found for visualization"
            }

    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics for the dataset"""
        stats = {}
        
        # Basic info
        stats['shape'] = f"{self.df.shape[0]} rows × {self.df.shape[1]} columns"
        stats['missing_values'] = self.df.isna().sum().to_dict()
        stats['duplicates'] = int(self.df.duplicated().sum())
        
        # Numeric columns statistics
        numeric_stats = {}
        for col in [col for col, typ in self.column_types.items() if typ == 'numeric']:
            numeric_stats[col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'std': self.df[col].std()
            }
        stats['numeric_stats'] = numeric_stats
        
        # Categorical columns statistics
        categorical_stats = {}
        for col in [col for col, typ in self.column_types.items() if typ == 'categorical']:
            categorical_stats[col] = {
                'unique_values': self.df[col].nunique(),
                'top_value': self.df[col].mode()[0] if not self.df[col].mode().empty else None,
                'top_value_count': self.df[col].value_counts().iloc[0] if not self.df[col].value_counts().empty else None
            }
        stats['categorical_stats'] = categorical_stats
        
        return {
            "type": "stats",
            "data": stats,
            "description": "Dataset Summary Statistics"
        }

    def create_agent(self) -> AgentExecutor:
        """Create LangChain agent if API key is provided"""
        if not self.api_key:
            return None
            
        tools = [
            Tool(
                name="auto_visualize",
                func=self.auto_visualize,
                description="Automatically generates appropriate visualizations based on the dataset"
            ),
            Tool(
                name="summary_stats",
                func=self.generate_summary_stats,
                description="Generates comprehensive statistics about the dataset"
            )
        ]
        
        return create_pandas_dataframe_agent(
            llm=self.llm,
            df=self.df,
            verbose=True,
            agent_type="openai-tools",
            max_iterations=7,
            early_stopping_method="generate",
            extra_tools=tools,
            memory=self.memory,
            allow_dangerous_code=True
        )