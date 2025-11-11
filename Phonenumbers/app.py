import streamlit as st
import requests
import pandas as pd
from io import StringIO

st.title("Image to CSV Table Extractor")

uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"File {i+1}: {uploaded_file.name}")
        
        url = "https://extraction-api.nanonets.com/extract"
        headers = {"Authorization": "Bearer 1171fb68-9df6-11f0-bab5-ba0abca6392e"}
        
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"output_type": "csv"}
        
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'content' in result:
                    phone_numbers = result['content'].replace(',', '\n').split('\n')
                    df = pd.DataFrame({'Phone Numbers': [num.strip() for num in phone_numbers if num.strip()]})
                    st.dataframe(df)
                    all_dfs.append(df)
                else:
                    st.json(result)
            else:
                st.error(f"Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
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