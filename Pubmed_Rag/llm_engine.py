import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pubmed_retriever import fetch_pubmed_abstracts

# Load our environment variables
load_dotenv()

# We use the 70B model here because it is highly capable of strictly following complex formatting instructions
llm = ChatGroq(
    temperature=0.1, # Very low temperature for factual consistency and minimal hallucination
    model_name="llama-3.3-70b-versatile" 
)

# This prompt strictly enforces your requested PDF template structure
CLINICAL_PROMPT = """
You are an expert clinical decision-support AI. Your job is to resolve clinical dilemmas using ONLY the provided recent PubMed literature.
You must output your response EXACTLY matching the markdown format below. Do not add conversational filler.

Here is the recent medical literature retrieved for the query:
{context}

Clinical Query: {query}

Respond EXACTLY in this format:

### CLINICAL BOTTOM LINE
[Direct answer in 2-3 sentences. This should answer the question immediately based on the context.]
**Evidence Strength:** [Rate from 1 to 5 stars, e.g., ★★★★☆ based on the provided text]
**Confidence:** [High/Medium/Low]
**Literature Match Score:** [Rate from 0% to 100% based strictly on how directly the retrieved abstracts answer the specific clinical query]

-
---
### WHY THIS MATTERS
[One paragraph explaining the clinical significance of this query based on the abstracts.]

---
### KEY TAKEAWAYS
✓ [Actionable Point 1]
✓ [Actionable Point 2]
✓ [Actionable Point 3]
✓ [Actionable Point 4]

---
### EVIDENCE SNAPSHOT
Current PubMed evidence suggests:
• [Main finding 1]
• [Main finding 2]
• [Main finding 3]

---
### LANDMARK STUDIES
**Study:** [Title of most relevant study from context]
**Population:** [Extract or infer patient population from abstract]
**Finding:** [Major finding]
**Clinical Impact:** [How this impacts practice]

---
### CLINICAL PEARLS
• [Important bedside insight based on the text]
• [Common mistake to avoid or monitoring consideration]
• [Key clinical reminder]

---
### LIMITATIONS OF EVIDENCE
• [Point out missing data, conflicting evidence, or uncertainties explicitly mentioned in the abstracts]

---
### EVIDENCE QUALITY
**Highest Level Evidence:** [e.g., RCT, Observational, Meta-analysis - infer from abstracts]
**Consistency:** [High/Medium/Low based on agreement between abstracts]

---
### PUBMED REFERENCES
[List the exact PMIDs from the provided context as clickable Markdown links using this format:]
- [PMID: Number](https://pubmed.ncbi.nlm.nih.gov/Number/)
"""

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=CLINICAL_PROMPT
)

# Create the LangChain pipeline
chain = prompt_template | llm

def generate_clinical_report(search_query, full_clinical_context=None):
    if full_clinical_context is None:
        full_clinical_context = search_query
    print(f"1. Fetching recent literature from PubMed for: {search_query}...")
    # Use the short keyword string for the API search
    papers = fetch_pubmed_abstracts(search_query, max_results=5)
    
    if not papers:
        return "No recent literature found for this query. Please try rephrasing."
        
    print("2. Formatting context for the LLM...")
    context_block = ""
    for paper in papers:
        context_block += f"PMID: {paper['pmid']}\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
        
    print("3. Generating structured clinical report via Groq...")
    # Feed the massive PDF text block to the LLM context
    response = chain.invoke({"context": context_block, "query": full_clinical_context})
    
    return response.content


# --- Test Block ---
if __name__ == "__main__":
    test_query = "Latest management of septic shock"
    final_report = generate_clinical_report(test_query)
    
    print("\n" + "="*50 + "\n")
    print(final_report)