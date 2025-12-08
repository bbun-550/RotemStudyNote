# pip install chromadb==0.5.23 --upgrade
# pip install sentence-transformers==3.2.1 chromadb==0.5.23

import chromadb
from chromadb import PersistentClient

client = PersistentClient(path=".chroma")
collection = client.get_or_create_collection("test")