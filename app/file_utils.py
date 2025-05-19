import chardet
import pandas as pd
import logging
from typing import Tuple

def detect_encoding(uploaded_file, sample_size: int = 100000) -> Tuple[str, float]:
    """Detect file encoding with error handling"""
    try:
        raw_data = uploaded_file.read(sample_size)
        uploaded_file.seek(0)
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']
    except Exception as e:
        logging.warning(f"Encoding detection failed: {str(e)}")
        return 'utf-8', 0.0

def load_data_file(uploaded_file) -> pd.DataFrame:
    """Robust data loader with automatic encoding detection"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try common encodings first
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='warn')
                except UnicodeDecodeError:
                    continue
            
            # Fallback to detected encoding
            encoding, _ = detect_encoding(uploaded_file)
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='warn')
        
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
            
    except Exception as e:
        raise ValueError(f"Could not load file: {str(e)}")