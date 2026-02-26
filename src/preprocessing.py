import pandas as pd
import numpy as np
import re
import urllib.parse
from sklearn.model_selection import train_test_split
import os

def parse_http_string(payload):
    """Shared logic to decompose raw HTTP snippets into standard fields."""
    row = {"method": "GET", "path": "/", "query": "", "headers": "", "body": ""}
    if not payload: return row
    
    # 1. Handle Method/Path pattern (e.g., 'GET /path?q=v {"body": 1}')
    if payload.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ')):
        parts = payload.split(' ', 2)
        row['method'] = parts[0]
        if len(parts) > 1:
            url_part = parts[1]
            try:
                p = urllib.parse.urlparse(url_part)
                row['path'] = p.path
                row['query'] = p.query
            except:
                row['path'] = url_part
        if len(parts) > 2:
            row['body'] = parts[2]
            
    # 2. Handle Body pattern (e.g., '{"key": "val"}')
    elif payload.startswith(('{', '[')):
        row['method'] = "POST"
        row['body'] = payload
        
    # 3. Handle Query pattern (e.g., 'id=1&name=test')
    elif any(sep in payload for sep in ['=', '&']) and ' ' not in payload:
        row['query'] = payload
    
    # 4. Fallback: Entire string as payload-carrying header/path
    else:
        row['path'] = payload
        
    return row

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase all text
    text = text.lower()
    
    # 2. URL decode (2 passes to handle double encoding)
    try:
        text = urllib.parse.unquote(text)
        text = urllib.parse.unquote(text)
    except:
        pass
    
    # 3. Normalize whitespace but keep everything else
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(input_path, output_dir):
    """Legacy preprocessing function - replaced by standardize_data.py but kept for compatibility."""
    df = pd.read_csv(input_path)
    df['cleaned_text'] = df.apply(lambda x: clean_text(str(x)), axis=1)
    # This is a stub for the old pipeline
    pass

if __name__ == "__main__":
    pass
