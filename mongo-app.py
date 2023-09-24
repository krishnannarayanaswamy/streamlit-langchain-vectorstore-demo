import pandas as pd
import streamlit as st
import dotenv
import langchain
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import BaseRetriever, Document, SystemMessage
from langchain.vectorstores import VectorStore

langchain.debug = True

dotenv.load_dotenv(dotenv.find_dotenv())

# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()

# name of the Redis search index to create
index_name = "products"

# assumes you have a redis stack server running on within your docker compose network
#redis_url = "redis://default:8jcCZoVHonkkNjCEAZsHXbQz9E63axK5@redis-17143.c299.asia-northeast1-1.gce.cloud.redislabs.com:17143"

# initialize MongoDB python client
client = MongoClient("mongodb+srv://demo:demo123@vectorsearchdemo.bhovs81.mongodb.net/?retryWrites=true&w=majority")

db_name = "mongovector"
collection_name = "demo"
collection = client[db_name][collection_name]
index_name = "_id_"
# create and load redis with documents
vectorstore = MongoDBAtlasVectorSearch(
    embedding=embedding,
    index_name=index_name,
    collection=collection
)


class MongoProductRetriever(BaseRetriever):
    vectorstore: VectorStore

    class Config:
        
        arbitrary_types_allowed = True

    def combine_metadata(self, doc) -> str:
        metadata = doc.metadata
        return (
            "Product name: " + metadata["product_name"] + ". " +
            "Item Description: " + metadata["description"] + ". " +
            "Item Price: " + metadata["price"] + "."
        )

    def get_relevant_documents(self, query):
        docs = []
        for doc in self.vectorstore.similarity_search(query):
            content = self.combine_metadata(doc)
            docs.append(Document(
                page_content=content,
                metadata=doc.metadata
            ))
        return docs

@st.cache_resource
def create_chatbot():
    print(f"Creating chatbot for ...")
    llm = ChatOpenAI(temperature=0, streaming=True)
    retriever = MongoProductRetriever(vectorstore=vectorstore)
    retriever_tool = create_retriever_tool(
        retriever, "products_retrevier", "Useful when searching for products from a product description. Prices are in USD.")
    system_message = "You are a customer service of a rtail eCommerce store and you are asked to pick products for a customer."
    message = SystemMessage(content=system_message)
    agent_executor = create_conversational_retrieval_agent(
        llm=llm, tools=[retriever_tool], system_message=message, verbose=True)
    return agent_executor


if 'history' not in st.session_state:
    st.session_state['history'] = []

st.set_page_config(layout="wide")

chatbot = create_chatbot()

# Display chat messages from history on app rerun
for (query, answer) in st.session_state['history']:
    with st.chat_message("User"):
        st.markdown(query)
    with st.chat_message("Bot"):
        st.markdown(answer)

prompt = st.chat_input(placeholder="Ask chatbot")
if prompt:
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("Bot"):
        st_callback = StreamlitCallbackHandler(st.container())
        result = result = chatbot.invoke({
            "input": prompt,
            "chat_history": st.session_state['history']
        }, config={"callbacks": [st_callback]})
        st.session_state['history'].append((prompt, result["output"]))
        st.markdown(result["output"])
