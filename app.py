import streamlit as st
import pandas as pd
from docstrange import DocumentExtractor
import tempfile
import os
from io import StringIO
import re

# Configure API key from secrets or environment
try:
    # Try to get from Streamlit secrets first
    api_key = st.secrets["docstrange"]["api_key"]
except:
    # Fallback to environment variable
    api_key = "3fef6743-96a4-11f0-8f02-d289544ddb4c"

os.environ['DOCSTRANGE_API_KEY'] = api_key

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

def clean_text(text):
    """Remove numbers and clean text for names and locations"""
    if not text:
        return ""
    # Remove standalone numbers but keep text with numbers (like MR, MS)
    words = text.split()
    cleaned_words = []
    for word in words:
        # Keep words that are not pure numbers and not phone/app ID patterns
        if not (word.isdigit() and len(word) >= 4):
            # Remove numbers from mixed text but keep letters
            cleaned_word = re.sub(r'\d+', '', word).strip()
            if cleaned_word and len(cleaned_word) > 1:
                cleaned_words.append(cleaned_word)
    return ' '.join(cleaned_words)

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
    
    # First extract all numbers for income (4+ digits)
    income_candidates = []
    filtered_parts = []
    for part in parts:
        if part.isdigit() and len(part) >= 4:
            income_candidates.append(part)
        else:
            filtered_parts.append(part)
    
    # Use the first income candidate
    if income_candidates:
        income = income_candidates[0]
    
    # Look for card type patterns (no numbers allowed)
    card_parts = []
    remaining_parts = []
    
    # Check for Ashva card (with fuzzy matching)
    ashva_found = False
    for i, part in enumerate(filtered_parts):
        part_upper = part.upper()
        # Fuzzy match for Ashva variations
        if ('ASHVA' in part_upper or 'ASTVA' in part_upper or 
            'ASHWA' in part_upper or 'ASCHVA' in part_upper or
            (len(part) >= 4 and part_upper.startswith('ASH') and 'V' in part_upper)):
            card_parts = [part]
            remaining_parts = filtered_parts[:i] + filtered_parts[i+1:]
            ashva_found = True
            break
    
    # If Ashva not found, look for other card patterns
    if not ashva_found:
        i = 0
        while i < len(filtered_parts):
            part = filtered_parts[i]
            if 'FIRST' in part.upper() or 'VISA' in part.upper() or 'RUPAY' in part.upper():
                # Take this part and next 2 parts for card type
                card_parts = filtered_parts[i:i+3] if i+2 < len(filtered_parts) else [part]
                remaining_parts = filtered_parts[:i] + filtered_parts[i+len(card_parts):]
                break
            i += 1
    
    if not card_parts:
        remaining_parts = filtered_parts
    
    # Clean card type (remove any numbers that might have slipped in)
    if card_parts:
        clean_card_parts = []
        for part in card_parts:
            # Remove numbers from card type parts
            clean_part = re.sub(r'\d+', '', part).strip()
            if clean_part:
                # Normalize Ashva variations
                part_upper = clean_part.upper()
                if ('ASHVA' in part_upper or 'ASTVA' in part_upper or 
                    'ASHWA' in part_upper or 'ASCHVA' in part_upper or
                    (len(clean_part) >= 4 and part_upper.startswith('ASH') and 'V' in part_upper)):
                    clean_card_parts.append('Ashva')
                else:
                    clean_card_parts.append(clean_part)
        card_type = ' '.join(clean_card_parts)
    
    # Look for data type
    final_parts = []
    for part in remaining_parts:
        if part.upper() in ['BUREAU', 'INCOME', 'CARD']:
            data_type = part
        else:
            final_parts.append(part)
    
    # If no income found yet, look in the remaining parts
    if not income:
        for part in final_parts[:]:
            if part.isdigit() and len(part) >= 4:
                income = part
                final_parts.remove(part)
                break
    
    # Remaining parts: first half as name, second half as location
    if final_parts:
        mid = len(final_parts) // 2
        name_raw = ' '.join(final_parts[:mid+1]) if final_parts else ""
        location_raw = ' '.join(final_parts[mid+1:]) if len(final_parts) > mid+1 else ""
        
        # Clean names and locations
        name = clean_text(name_raw)
        location = clean_text(location_raw)
    
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
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'all_data' not in st.session_state:
    st.session_state.all_data = []
if 'failed_files' not in st.session_state:
    st.session_state.failed_files = []

# Auto-backup to browser storage
if st.session_state.all_data:
    backup_data = pd.concat(st.session_state.all_data, ignore_index=True, sort=False)
    st.session_state['backup_csv'] = backup_data.to_csv(index=False)

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

# Data management buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear All Data", type="secondary"):
        st.session_state.processed_files = set()
        st.session_state.all_data = []
        st.session_state.failed_files = []
        if 'backup_csv' in st.session_state:
            del st.session_state['backup_csv']
        st.success("All data cleared!")
        st.rerun()

with col2:
    if 'backup_csv' in st.session_state and st.session_state.backup_csv:
        st.download_button(
            label="💾 Download Backup",
            data=st.session_state.backup_csv,
            file_name="backup_data.csv",
            mime="text/csv",
            key="download_backup"
        )

# Show current progress
if st.session_state.all_data:
    st.info(f"📊 Current Progress: {len(st.session_state.processed_files)} files processed, {sum(len(df) for df in st.session_state.all_data)} total rows")

uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)

if uploaded_files:
    new_files_processed = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Create unique file identifier
        file_id = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
        
        # Skip if already processed
        if file_id in st.session_state.processed_files:
            st.info(f"File {uploaded_file.name} already processed - skipping")
            continue
            
        st.subheader(f"File {i+1}: {uploaded_file.name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Initialize extractor with explicit API key
            extractor = DocumentExtractor(api_key=api_key)
            
            # Add progress indicator
            progress_placeholder = st.empty()
            progress_placeholder.info(f"🔄 Processing {uploaded_file.name}...")
            
            try:
                result = extractor.extract(tmp_path)
                progress_placeholder.success(f"✅ File loaded: {uploaded_file.name}")
            except Exception as extract_error:
                error_msg = str(extract_error).lower()
                if any(keyword in error_msg for keyword in ['network', 'connection', 'timeout', 'unreachable', 'dns']):
                    st.error(f"🌐 Network error for {uploaded_file.name}: Connection interrupted. Skipping to next file.")
                    st.session_state.failed_files.append(f"{uploaded_file.name} - Network Error")
                    progress_placeholder.empty()
                    continue
                else:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {extract_error}")
                    progress_placeholder.empty()
                    continue
            
            # Try different extraction methods with better error handling
            csv_content = None
            
            try:
                progress_placeholder.info(f"📊 Extracting data from {uploaded_file.name}...")
                csv_content = result.extract_csv()
                st.success(f"CSV extraction successful for {uploaded_file.name}")
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['network', 'connection', 'timeout', 'unreachable']):
                    st.error(f"🌐 Network broke during CSV extraction for {uploaded_file.name}. Moving to next file.")
                    progress_placeholder.empty()
                    continue
                
                st.warning(f"CSV extraction failed: {str(e)}")
                try:
                    progress_placeholder.info(f"📝 Trying text extraction for {uploaded_file.name}...")
                    text_content = result.extract_text()
                    if text_content and text_content.strip():
                        st.info(f"Text extraction successful for {uploaded_file.name}")
                        # Convert text to CSV format
                        lines = text_content.strip().split('\n')
                        csv_lines = []
                        for line in lines:
                            if line.strip():
                                parts = re.split(r'\s{2,}', line.strip())
                                if len(parts) == 1:
                                    parts = line.strip().split()
                                csv_lines.append(','.join(parts))
                        csv_content = '\n'.join(csv_lines)
                    else:
                        st.error(f"No content extracted from {uploaded_file.name}")
                        progress_placeholder.empty()
                        continue
                except Exception as text_error:
                    error_msg = str(text_error).lower()
                    if any(keyword in error_msg for keyword in ['network', 'connection', 'timeout']):
                        st.error(f"🌐 Network broke during text extraction for {uploaded_file.name}. Moving to next file.")
                        progress_placeholder.empty()
                        continue
                    st.error(f"All extraction methods failed: {str(text_error)}")
                    progress_placeholder.empty()
                    continue
            
            progress_placeholder.empty()
            
            if not csv_content or csv_content.strip() == "":
                st.warning(f"No extractable content found in {uploaded_file.name}")
                continue
            
            # Show debug info
            st.write(f"**Raw CSV content (first 500 chars):**")
            st.text(csv_content[:500] if csv_content else "No content")
            
            # Parse CSV with error handling
            try:
                df = pd.read_csv(StringIO(csv_content), header=None, sep=None, engine='python')
                if df.empty:
                    df = pd.read_csv(StringIO(csv_content), header=None)
            except pd.errors.EmptyDataError:
                st.warning(f"No data found in {uploaded_file.name}")
                continue
            except Exception as csv_error:
                try:
                    lines = csv_content.strip().split('\n')
                    data = []
                    for line in lines:
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
            
            if not formatted_df.empty:
                # Add to session state
                st.session_state.all_data.append(formatted_df)
                st.session_state.processed_files.add(file_id)
                new_files_processed.append(formatted_df)
                
                # Show combined data after each file
                current_combined_df = pd.concat(st.session_state.all_data, ignore_index=True, sort=False)
                st.write(f"**Combined Data (Files 1-{len(st.session_state.all_data)}):**")
                st.write(f"Total rows so far: {len(current_combined_df)}")
                st.dataframe(current_combined_df)
                
                # Auto-save progress
                st.download_button(
                    label=f"💾 Download Progress ({len(st.session_state.all_data)} files)",
                    data=current_combined_df.to_csv(index=False),
                    file_name=f"progress_{len(st.session_state.all_data)}_files.csv",
                    mime="text/csv",
                    key=f"download_progress_{file_id}"
                )
                
            else:
                st.warning(f"No valid data found in {uploaded_file.name} after validation")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Show results if we have any data
    if st.session_state.all_data:
        # Combine all data from session state
        final_combined_df = pd.concat(st.session_state.all_data, ignore_index=True, sort=False)
        
        st.subheader("Final Combined Data")
        st.write(f"**Application Type:** {selected_app_type}")
        st.write(f"**Total files processed:** {len(st.session_state.processed_files)}")
        st.write(f"**Total rows:** {len(final_combined_df)}")
        st.write(f"**Total columns:** {len(final_combined_df.columns)}")
        
        if new_files_processed:
            st.success(f"✅ {len(new_files_processed)} new files processed in this session")
        
        # Show failed files if any
        if st.session_state.failed_files:
            with st.expander(f"⚠️ Failed Files ({len(st.session_state.failed_files)})"):
                for failed_file in st.session_state.failed_files:
                    st.write(f"- {failed_file}")
        
        # Split data for Fresh Incomplete Application
        if selected_app_type == "Fresh Incomplete Application":
            total_rows = len(final_combined_df)
            mid_point = total_rows // 2
            
            # D gets extra row if odd number
            df_p = final_combined_df.iloc[:mid_point].copy()
            df_d = final_combined_df.iloc[mid_point:].copy()
            
            # Update Type of Data column
            df_p['Type of Data'] = 'Fresh Incomplete Application P'
            df_d['Type of Data'] = 'Fresh Incomplete Application D'
            
            st.write(f"**Split Data:**")
            st.write(f"- Fresh Incomplete Application P: {len(df_p)} rows")
            st.write(f"- Fresh Incomplete Application D: {len(df_d)} rows")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Fresh Incomplete Application P:**")
                st.dataframe(df_p)
                st.download_button(
                    label="Download P Data",
                    data=df_p.to_csv(index=False),
                    file_name="fresh_incomplete_P.csv",
                    mime="text/csv",
                    key="download_p_data"
                )
            
            with col2:
                st.write("**Fresh Incomplete Application D:**")
                st.dataframe(df_d)
                st.download_button(
                    label="Download D Data",
                    data=df_d.to_csv(index=False),
                    file_name="fresh_incomplete_D.csv",
                    mime="text/csv",
                    key="download_d_data"
                )
            
            # Combined download
            combined_split_df = pd.concat([df_p, df_d], ignore_index=True)
            st.download_button(
                label="Download Combined P+D Data",
                data=combined_split_df.to_csv(index=False),
                file_name="fresh_incomplete_combined.csv",
                mime="text/csv",
                key="download_combined_split"
            )
        else:
            st.dataframe(final_combined_df)
            # Download button for other types
            st.download_button(
                label="Download Final CSV",
                data=final_combined_df.to_csv(index=False),
                file_name="data1024.csv",
                mime="text/csv",
                key="download_final_csv"
            )