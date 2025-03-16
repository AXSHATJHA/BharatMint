import pandas as pd
import chromadb

df_cases = pd.read_csv("lawyergpt_db.csv")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)



