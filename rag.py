# %%
import os
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


from langchain_text_splitters import RecursiveCharacterTextSplitter


# %%
def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified directory path.

    This function supports loading of PDF, Markdown, and HTML documents by utilizing
    different loaders for each file type. It checks if the provided path exists and
    raises a FileNotFoundError if it does not. It then iterates over the supported
    file types and uses the corresponding loader to load the documents into a list.

    Args:
        path (str): The path to the directory containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs


# %%

documents = load_documents(path="docs")

# %%
txt = "////".join([x.page_content for x in documents])

# %%
txt = txt.replace("\n", " ")

# %%
txt

# %% [markdown]
# # Split documents
#
# Split documents into small chunks. This is so we can find the most relevant chunks for a query and pass only those into the LLM.

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=50, separators=["\n\n"]
)
texts = text_splitter.split_documents(documents)

# %%
len(texts)


# %%
texts[100]

# %% [markdown]
# # Initialize ChromaDB
#
# Create embeddings for each chunk and insert into the Chroma vector database.
#

# %%
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db4")


# %%
docs = vectordb.similarity_search(
    "was ist die Voraussetzung eine Blaue Karte zu bekommen?"
)
docs

# %% [markdown]
# # RAG

# %%
# LLM prompt template
template = """You are an assistant for specific knowledge query tasks. 
   Use the following pieces of retrieved context to answer the question. 
   If you don't know the answer, just say that you don't know. 
   Question: {question} 
   Context: {context} 
   Answer:
   """

# RAG prompt
template1 = """Answer the question based only on the following context:
{context}
Question: {question}
"""

# %%
# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters

llm = ChatOllama(model="phi3")


prompt = ChatPromptTemplate.from_template(template1)


# %%
retriever = vectordb.as_retriever(k=3)


# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# %%
# RAG

chain = (
    RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    | prompt
    | llm
    | StrOutputParser()
)

# see link below how to return the sources
# https://python.langchain.com/v0.1/docs/use_cases/question_answering/sources/

# %%

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


# %%

res = chain.invoke("was ist die Voraussetzung eine Blaue Karte zu bekommen?")

# %%
res["answer"]
