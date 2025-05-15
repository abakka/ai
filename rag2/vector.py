from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("sentimentdataset.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chorma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content = row["Text"] + " " + row["Hashtags"],
            metadata = {"sentiment": row["Sentiment"],
                        "timestamp": row["Timestamp"], 
                        "platform": row["Platform"],
                        "likes": row["Likes"],
                        "retweets": row["Retweets"],
                        "country": row["Country"],
                        "user": row["User"],
                        "year": row["Year"],
                        "month": row["Month"],
                        "day": row["Day"],
                        "hour": row["Hour"]},
            id=str(i)
            )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="sentiment_posts_dataset",
    persist_directory=db_location,
    embedding_function=embeddings
    )

if add_documents:
    vector_store.add_documents(documents=documents,ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k" : 3}
    )