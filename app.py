import streamlit as st
import pandas as pd
from docstrange import DocumentExtractor
import tempfile
import os

# Configure API key
os.environ['DOCSTRANGE_API_KEY'] = "3fef6743-96a4-11f0-8f02-d289544ddb4c"

st.title("Image to CSV Table Extractor")

uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"File {i+1}: {uploaded_file.name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            extractor = DocumentExtractor()
            result = extractor.extract(tmp_path)
            csv_content = result.extract_csv()
            
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            st.dataframe(df)
            all_dfs.append(df)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            os.unlink(tmp_path)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        st.subheader("Combined Data")
        st.dataframe(combined_df)
        
        st.download_button(
            label="Download Combined CSV",
            data=combined_df.to_csv(index=False),
            file_name="combined_tables.csv",
            mime="text/csv"
        )