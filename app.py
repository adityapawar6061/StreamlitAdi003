import streamlit as st
import pandas as pd
from docstrange import DocumentExtractor
import tempfile
import os
from io import StringIO

# Configure API key
os.environ['DOCSTRANGE_API_KEY'] = "3fef6743-96a4-11f0-8f02-d289544ddb4c"

# Load language mapping
@st.cache_data
def load_language_mapping():
    """Load language mapping from Languages.csv"""
    try:
        lang_df = pd.read_csv('Languages.csv')
        # Create mapping from location to language
        location_to_language = {}
        for _, row in lang_df.iterrows():
            if pd.notna(row['DISTRICT NAME']):
                location_to_language[row['DISTRICT NAME'].upper()] = row['Language']
        return location_to_language
    except:
        return {}

def get_language_for_location(location, location_mapping):
    """Get language for a given location"""
    if not location or pd.isna(location):
        return "Hindi"  # Default
    
    location_upper = str(location).upper()
    
    # Direct match
    if location_upper in location_mapping:
        return location_mapping[location_upper]
    
    # Partial match
    for loc_key, lang in location_mapping.items():
        if location_upper in loc_key or loc_key in location_upper:
            return lang
    
    return "Hindi"  # Default fallback

import re

def extract_phone_number(text):
    """Extract 10-digit phone number from text"""
    if not text:
        return "", text
    
    # Find 10-digit numbers
    phone_pattern = r'\b\d{10}\b'
    matches = re.findall(phone_pattern, str(text))
    
    if matches:
        phone = matches[0]
        # Remove phone from original text
        cleaned_text = re.sub(phone_pattern, '', str(text)).strip()
        return phone, cleaned_text
    
    return "", text

def extract_application_id(text):
    """Extract 9-digit application ID from text"""
    if not text:
        return "", text
    
    # Find 9-digit numbers
    app_id_pattern = r'\b\d{9}\b'
    matches = re.findall(app_id_pattern, str(text))
    
    if matches:
        app_id = matches[0]
        # Remove app_id from original text
        cleaned_text = re.sub(app_id_pattern, '', str(text)).strip()
        return app_id, cleaned_text
    
    return "", text

def clean_and_separate_data(row_values):
    """Clean and separate combined data"""
    all_text = ' '.join([str(val) for val in row_values if val])
    
    # Extract application ID
    app_id, remaining_text = extract_application_id(all_text)
    
    # Extract phone number
    phone, remaining_text = extract_phone_number(remaining_text)
    
    # Split remaining text for name and location
    parts = [part.strip() for part in remaining_text.split() if part.strip()]
    
    # Try to identify name (usually first few words) and location
    name = ""
    location = ""
    card_type = ""
    income = ""
    data_type = ""
    
    # Look for common patterns
    for i, part in enumerate(parts):
        if 'FIRST' in part.upper() or 'VISA' in part.upper() or 'RUPAY' in part.upper():
            card_type = ' '.join(parts[i:i+3]) if i+2 < len(parts) else part
            parts = parts[:i] + parts[i+3:]
            break
    
    # Look for income (numbers)
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) >= 4:
            income = part
            parts.pop(i)
            break
    
    # Look for data type
    for i, part in enumerate(parts):
        if part.upper() in ['BUREAU', 'INCOME', 'CARD']:
            data_type = part
            parts.pop(i)
            break
    
    # Remaining parts: first half as name, second half as location
    if parts:
        mid = len(parts) // 2
        name = ' '.join(parts[:mid+1]) if parts else ""
        location = ' '.join(parts[mid+1:]) if len(parts) > mid+1 else ""
    
    return app_id, name, phone, location, card_type, income, data_type

def format_to_target_structure(df, app_type, file_name, location_mapping):
    """Format dataframe to match target CSV structure"""
    if df.empty:
        return pd.DataFrame()
    
    target_columns = [
        'Type of Data', 'Language', 'ApplicationID', 'Location', 
        'Card Type', 'Income', 'Data Type', 'Name', 'Phone Number', 'User', 'FileName'
    ]
    
    formatted_df = pd.DataFrame()
    
    for i, row in df.iterrows():
        row_values = [str(val) if pd.notna(val) and str(val).strip() else "" for val in row.values]
        
        # Clean and separate data
        app_id, name, phone, location, card_type, income, data_type = clean_and_separate_data(row_values)
        
        # Different validation based on application type
        if app_type == "Fresh Incomplete Application":
            # Must have both ApplicationID and Phone Number
            if not (app_id and len(app_id) == 9 and phone and len(phone) == 10):
                continue
        else:  # "Verification Rejection" or "Already IDFC Carded"
            # Only phone number is compulsory
            if not (phone and len(phone) == 10):
                continue
        
        new_row = {
            'Type of Data': app_type,
            'ApplicationID': app_id if app_id else "NULL",
            'Name': name if name else "NULL",
            'Phone Number': phone,
            'Location': location if location else "NULL",
            'Card Type': card_type if card_type else "NULL",
            'Income': income if income else "NULL",
            'Data Type': data_type if data_type else "NULL",
            'User': "NULL",
            'FileName': file_name
        }
        
        # Get language based on location
        new_row['Language'] = get_language_for_location(location, location_mapping)
        
        formatted_df = pd.concat([formatted_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Ensure all target columns exist
    for col in target_columns:
        if col not in formatted_df.columns:
            formatted_df[col] = "NULL"
    
    return formatted_df[target_columns]

st.title("Image to CSV Table Extractor")

# Load language mapping
location_mapping = load_language_mapping()

# Initialize session state
if 'app_type' not in st.session_state:
    st.session_state.app_type = "Fresh Incomplete Application"

# Application type selection
st.subheader("Select Application Type")
app_type_options = [
    "Fresh Incomplete Application",
    "Already IDFC Carded", 
    "Verification Rejection"
]

selected_app_type = st.selectbox(
    "Choose the application type that will be applied to all extracted data:",
    app_type_options,
    index=app_type_options.index(st.session_state.app_type),
    key="app_type_selector"
)

# Update session state
st.session_state.app_type = selected_app_type

st.write(f"**Selected Type:** {selected_app_type}")

# Show validation rules
if selected_app_type == "Fresh Incomplete Application":
    st.info("📋 Validation: Both Application ID (9 digits) and Phone Number (10 digits) required")
else:
    st.info("📋 Validation: Only Phone Number (10 digits) required. Application ID optional.")

uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)

if uploaded_files:
    all_formatted_dfs = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"File {i+1}: {uploaded_file.name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            extractor = DocumentExtractor()
            result = extractor.extract(tmp_path)
            csv_content = result.extract_csv()
            
            # Parse CSV with error handling - capture ALL columns
            try:
                # Try multiple parsing methods to capture all data
                df = pd.read_csv(StringIO(csv_content), header=None, sep=None, engine='python')
                if df.empty:
                    df = pd.read_csv(StringIO(csv_content), header=None)
            except pd.errors.EmptyDataError:
                st.warning(f"No data found in {uploaded_file.name}")
                continue
            except Exception as csv_error:
                # Try reading as raw text and splitting
                try:
                    lines = csv_content.strip().split('\n')
                    data = []
                    for line in lines:
                        # Split by common delimiters
                        row = re.split(r'[,\t\|;]', line)
                        data.append(row)
                    df = pd.DataFrame(data)
                except:
                    st.error(f"Could not parse CSV from {uploaded_file.name}: {csv_error}")
                    continue
            
            # Format to target structure
            formatted_df = format_to_target_structure(df, selected_app_type, uploaded_file.name, location_mapping)
            
            st.write(f"**Original extracted data:**")
            st.dataframe(df)
            st.write(f"**Formatted data:**")
            st.dataframe(formatted_df)
            
            all_formatted_dfs.append(formatted_df)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    if all_formatted_dfs:
        # Combine all formatted dataframes
        final_combined_df = pd.concat(all_formatted_dfs, ignore_index=True, sort=False)
        
        st.subheader("Final Combined Data")
        st.write(f"**Application Type:** {selected_app_type}")
        st.write(f"**Total rows:** {len(final_combined_df)}")
        st.write(f"**Total columns:** {len(final_combined_df.columns)}")
        st.dataframe(final_combined_df)
        
        # Download button
        st.download_button(
            label="Download Final CSV",
            data=final_combined_df.to_csv(index=False),
            file_name=f"extracted_data_{selected_app_type.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            key="download_final_csv"
        )