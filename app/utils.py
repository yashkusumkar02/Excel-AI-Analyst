import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from typing import Dict, Any

class DataAnalyzer:
    @staticmethod
    def comprehensive_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary with visualizations"""
        analysis = {}
        buffer = BytesIO()
        
        # Basic metadata
        analysis["Metadata"] = {
            "Dimensions": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "Memory Usage": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
            "Columns": [{"name": col, "type": str(df[col].dtype)} for col in df.columns]
        }
        
        # Data quality analysis
        analysis["Data Quality"] = {
            "Missing Values": df.isna().sum().to_dict(),
            "Duplicate Rows": int(df.duplicated().sum()),
            "Cardinality": {col: df[col].nunique() for col in df.columns}
        }
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            analysis["Numeric Analysis"] = {
                "Descriptive Stats": df[numeric_cols].describe().to_dict(),
                "Correlation Matrix": df[numeric_cols].corr().to_dict()
            }
            
            # Generate distribution plots
            try:
                plt.figure(figsize=(12, 8))
                df[numeric_cols].hist(bins=15, layout=(-1, 3))
                plt.tight_layout()
                plt.savefig(buffer, format='png')
                plt.close()
                analysis["Visualizations"] = {
                    "Distributions": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
                }
            except Exception as e:
                analysis["Visualizations"] = {"Error": str(e)}
        
        return analysis
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect anomalies using Z-score method"""
        numeric_cols = df.select_dtypes(include=np.number).columns
        anomalies = {}
        
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            anomalies[col] = {
                "count": int(sum(z_scores > threshold)),
                "indices": df.index[z_scores > threshold].tolist(),
                "threshold": threshold
            }
        
        return {"Anomalies": anomalies}