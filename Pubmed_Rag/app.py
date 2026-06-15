import streamlit as st
import re
from collections import Counter
import PyPDF2
from llm_engine import generate_clinical_report

def extract_top_phrases(text, num_phrases=2):
    """Extracts the most frequent 2-word phrases (bigrams), filtering out junk data."""
    # Added "pmid" and a few other generic terms to the stop words
    stop_words = {"this", "that", "with", "from", "your", "what", "when", "where", "how", "have", "has", "been", "would", "could", "should", "their", "there", "about", "which", "these", "those", "patient", "patients", "study", "treatment", "clinical", "evidence", "finding", "example", "early", "studies", "available", "data", "using", "pmid", "review"}
    
    # Extract words (4+ letters)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Filter out stop words AND words that are just repeating 'x's (like 'xxxxxxxx')
    filtered_words = [w for w in words if w not in stop_words and not re.match(r'^x+$', w)]
    
    if len(filtered_words) < 2:
        return []

    # Create bigrams
    bigrams = [f"{filtered_words[i]} {filtered_words[i+1]}" for i in range(len(filtered_words)-1)]
    
    # Count frequencies and grab the top results
    most_common = Counter(bigrams).most_common(num_phrases)
    
    return [phrase for phrase, count in most_common]



# 1. Page Config
st.set_page_config(
    page_title="Clinical Evidence Mediator",
    page_icon="⚕️",
    layout="centered"
)

# 2. Inject Custom CSS for Fonts, Animations, and Layout Tuning
st.markdown("""
    <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .main {
            animation: fadeIn 0.8s ease-in-out;
            font-family: 'Inter', sans-serif;
        }
        h1 {
            color: #0056b3;
            font-weight: 700;
        }
        /* Make the columns align nicely at the bottom */
        div[data-testid="column"] {
            display: flex;
            align-items: flex-end;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Header
st.title("⚕️ Clinical Evidence Mediator")
st.markdown("Enter a clinical dilemma below. You can also attach a reference PDF to include it in the analysis.")

# Initialize session state to hold PDF text behind the scenes
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = ""

st.divider()

# 4. SINGLE BAR UI: [ + Button ] [ Text Input ] [ Submit Button ]
col1, col2, col3 = st.columns([1, 6, 2])

with col1:
    # st.popover creates a clean button that opens a menu when clicked
    with st.popover("➕ PDF"):
        uploaded_file = st.file_uploader("Attach PDF", type="pdf", label_visibility="collapsed")
        if uploaded_file is not None:
            # Extract inline silently
            reader = PyPDF2.PdfReader(uploaded_file)
            st.session_state.pdf_context = "".join([page.extract_text() + "\n" for page in reader.pages])
            st.success("Attached!")

with col2:
    query = st.text_input(
        "Query", 
        placeholder="e.g., Latest management of septic shock...", 
        label_visibility="collapsed" # Hides the label so it sits perfectly inline
    )

with col3:
    submit = st.button("Generate Report", use_container_width=True, type="primary")

# 5. Process the Single Bar Submission
if submit:
    if not query.strip():
        st.warning("Please enter a clinical query to proceed.")
    else:
        with st.spinner("Analyzing literature and generating inline report..."):
            try:
                # 1. The Search Query is STRICTLY the text box input
                search_query = query
                
                # 2. The LLM Context bundles the text box AND the PDF (if attached)
                full_clinical_context = query
                if st.session_state.pdf_context:
                    # Append up to 3000 characters of the PDF for the LLM to read
                    full_clinical_context = f"{query}\n\n[USER PROVIDED PDF CONTEXT]:\n{st.session_state.pdf_context[:3000]}"
                
                # 3. Generate Report using the decoupled variables
                final_report = generate_clinical_report(search_query, full_clinical_context)
                
                # Display Results
                st.success(f"Report Generated! (Searched PubMed cleanly for: `{search_query}`)")
                st.markdown(final_report)
                
                # Save the state to show the feedback widget
                st.session_state.report_generated = True

            except Exception as e:
                st.error(f"An error occurred: {e}")
            


# 6. Feedback Mechanism
if st.session_state.get("report_generated", False):
    st.divider()
    st.markdown("**Was this consensus helpful?**")
    feedback = st.feedback("thumbs")
    
    if feedback is not None:
        if feedback == 1:
            st.toast("Thank you for the positive feedback!", icon="👍")
        elif feedback == 0:
            st.toast("Thank you for the feedback. We will work to improve accuracy.", icon="👎")