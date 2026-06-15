import streamlit as st
from llm_engine import generate_clinical_report

# 1. Configure the page settings
st.set_page_config(
    page_title="Clinical Evidence Mediator",
    page_icon="⚕️",
    layout="centered"
)

# 2. Build the Header
st.title("⚕️ Clinical Evidence Mediator")
st.markdown("""
Enter a clinical dilemma or medical question below. 
The system will search the latest PubMed literature (last 2 years) and generate an evidence-based, structured clinical consensus.
""")

# 3. Create the Input Field
query = st.text_input(
    "Clinical Query:", 
    placeholder="e.g., Latest management of septic shock"
)

# 4. Handle the Button Click
if st.button("Generate Clinical Report"):
    if not query.strip():
        st.warning("Please enter a clinical query to proceed.")
    else:
        # Show a loading spinner while the backend does the heavy lifting
        with st.spinner("Fetching recent literature and synthesizing evidence..."):
            try:
                # Call our RAG engine
                final_report = generate_clinical_report(query)
                
                # Render the markdown report
                st.success("Report Generated Successfully!")
                st.markdown(final_report)
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")