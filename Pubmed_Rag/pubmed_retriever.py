import os
from Bio import Entrez
from dotenv import load_dotenv

# Load our environment variables
load_dotenv()

# NCBI requires an email address to track API usage. 
Entrez.email = os.getenv("ENTREZ_EMAIL")

def fetch_pubmed_abstracts(query, max_results=5):
    """
    Searches PubMed for a query, filters for the last 2 years, 
    and retrieves the abstracts.
    """
    print(f"Searching PubMed for: '{query}'...")
    
    try:
        # Step 1: Search for the IDs (PMIDs) of relevant articles
        # reldate=730 restricts the search to the last ~2 years (730 days)
        search_handle = Entrez.esearch(
            db="pubmed", 
            term=query, 
            retmax=max_results, 
            sort="relevance",
            reldate=730 
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results["IdList"]
        
        if not id_list:
            print("No articles found.")
            return []

        # Step 2: Fetch the actual details for those PMIDs
        fetch_handle = Entrez.efetch(
            db="pubmed", 
            id=id_list, 
            retmode="xml"
        )
        papers = Entrez.read(fetch_handle)
        fetch_handle.close()

        # Step 3: Parse the XML response into a clean list of dictionaries
        formatted_results = []
        for paper in papers['PubmedArticle']:
            pmid = str(paper['MedlineCitation']['PMID'])
            article = paper['MedlineCitation']['Article']
            title = article.get('ArticleTitle', 'No title available')
            
            # Abstracts can be split into multiple sections in PubMed, so we join them
            abstract_text = ""
            if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                abstract_text = " ".join(article['Abstract']['AbstractText'])
            else:
                abstract_text = "No abstract available."

            formatted_results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract_text
            })

        return formatted_results

    except Exception as e:
        print(f"An error occurred while fetching from PubMed: {e}")
        return []

# --- Test Block ---
# This only runs if we execute this file directly, not when we import it later.
if __name__ == "__main__":
    test_query = "Latest management of septic shock"
    results = fetch_pubmed_abstracts(test_query, max_results=2)
    
    print("\n--- Test Results ---")
    for res in results:
        print(f"PMID: {res['pmid']}")
        print(f"Title: {res['title']}")
        print(f"Abstract Snippet: {res['abstract'][:150]}...\n")