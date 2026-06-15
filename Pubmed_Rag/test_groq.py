import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load variables from the .env file
load_dotenv()

print("Testing Groq API Connection...")

try:
    # Initialize the Llama 3 8B model for a fast test
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant" 
    )

    # Send a test prompt
    response = llm.invoke("You are a medical assistant. Say 'System Online' if you can hear me.")
    print(f"Response: {response.content}")

except Exception as e:
    print(f"Connection failed. Error: {e}")