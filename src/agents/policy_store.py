"""
AusClaim AI — Policy Store (Module 6)
ChromaDB vector store for AusClaim insurance policy rules.
Replaces hardcoded knowledge base in research_policy node.
"""

import chromadb
import uuid
import os

BASE_DIR = os.path.dirname(__file__)

# Initialize the ChromaDB client and create a collection for storing policy rules
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="ausclaim_policies")

# read the policy rules from the file and add them to the collection
with open(os.path.join(BASE_DIR, "policies.txt"), "r") as f:
    policies: list[str] = f.read().splitlines()

collection.add(ids=[str(uuid.uuid4()) for _ in policies], documents=policies)


# Query the collection for a specific claim type and return the corresponding policy rule
def query_policy(claim_type: str) -> str:
    query = f"What are the requirements for a {claim_type} insurance claim?"
    if claim_type == "other":
        query = "What happens when a claim does not match any standard category?"
    result = collection.query(
        query_texts=[query],
        n_results=1,
    )
    return result["documents"][0][0]


# Test the query_policy function with different claim types
if __name__ == "__main__":
    test_claims = ["motor_vehicle", "property", "public_liability", "other"]
    for claim in test_claims:
        print(f"\n{claim}: {query_policy(claim)}")
