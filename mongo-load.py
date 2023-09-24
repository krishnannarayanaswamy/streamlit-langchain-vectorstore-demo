import openai
import numpy
import pandas as pd
import streamlit as st
import dotenv
import langchain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

import json

from langchain.schema import BaseRetriever
from langchain.vectorstores import VectorStore
from langchain.schema import Document
from pydantic import BaseModel

langchain.debug = True

dotenv.load_dotenv(dotenv.find_dotenv())
MAX_TEXT_LENGTH=1000  # Maximum num of text characters to use
 
def auto_truncate(val):
 
    """Truncate the given text."""
 
    return val[:MAX_TEXT_LENGTH]
 
# Load Product data and truncate long text fields
 
all_prods_df = pd.read_csv("ProductDataset.csv", converters={
 
    #'bullet_point': auto_truncate,
    'product_name': auto_truncate,
    'description': auto_truncate
 
    #'item_keywords': auto_truncate,
 
    #'item_name': auto_truncate
 
})

# Replace empty strings with None and drop
 
#all_prods_df['item_keywords'].replace('', None, inplace=True)
all_prods_df['description'].replace('', None, inplace=True)
all_prods_df['price'].replace('', None, inplace=True)
all_prods_df['price'].replace('$', '', inplace=True)
convert_to_string = lambda x: str(x)
all_prods_df['price'].map(convert_to_string)
#all_prods_df.dropna(subset=['item_keywords'], inplace=True)
all_prods_df.dropna(subset=['description'], inplace=True)
 
# Reset pandas dataframe index
 
all_prods_df.reset_index(drop=True, inplace=True)

# Num products to use (subset)
NUMBER_PRODUCTS = 2500  
 
# Get the first 2500 products
product_metadata = ( 
    all_prods_df
     .head(NUMBER_PRODUCTS)
     .to_dict(orient='index')
)
 
# Check one of the products

print(product_metadata[0])

# data that will be embedded and converted to vectors
texts = [
     f"{v['description']} price: {v['price']}"  for k, v in product_metadata.items()
]
print(texts)
# product metadata that we'll store along our vectors
metadatas = list(product_metadata.values())

# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()


# assumes you have a redis stack server running on within your docker compose network
#redis_url = "redis://default:8jcCZoVHonkkNjCEAZsHXbQz9E63axK5@redis-17143.c299.asia-northeast1-1.gce.cloud.redislabs.com:17143"

# initialize MongoDB python client
client = MongoClient("mongodb+srv://demo:demo123@vectorsearchdemo.bhovs81.mongodb.net/?retryWrites=true&w=majority")

db_name = "mongovector"
collection_name = "demo"
collection = client[db_name][collection_name]
index_name = "_id_"
# create and load redis with documents
vectorstore = MongoDBAtlasVectorSearch.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embedding,
    index_name=index_name,
    collection=collection
)

#vectorstore.write_schema("redis_schema.yaml")
