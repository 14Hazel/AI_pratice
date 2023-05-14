import click
import os.path, os, sys
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


@click.command(["-h, --help"])
@click.option('--query', required=True, help="input your question")
@click.option('--file', required=True, help="Enter knowledge db path")
def main(query, file):

    loader = doc_loader(file)
    docs = construct_docs(loader)
    documents = split_text(docs, chunk_size=2000)
    vecstore = build_vector_db(documents)
    memory = memory_mode()
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0),
                                           vecstore.as_retriever(), memory=memory)
    result = qa({"question": query})
    print(result['answer'])

def doc_loader(filepath):
    if os.path.isdir(filepath):
        file_list = os.listdir(filepath)
        loader = [UnstructuredHTMLLoader(os.path.join(filepath,file)) for file in file_list]
        return loader
    elif os.path.isfile(filepath):
        loader = UnstructuredHTMLLoader(filepath)
        return loader
    else:
        raise FileNotFoundError
    
      
def construct_docs(loader):
    loaders = loader
    if isinstance(loader, list):
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
    else:
        docs = loaders.load()
    return docs

def split_text(docs, chunk_size):
    """
    return split docs
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, 
                                          chunk_overlap=0)
    documents = text_splitter.split_documents(docs)

    return documents

def build_vector_db(documents):
    embeddings = OpenAIEmbeddings()
    vecstore = Chroma.from_documents(documents, embeddings)
    return vecstore

# introduce memory mechanism so it would consume token again and again

def memory_mode():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory





        



if __name__=='__main__':
    main()
