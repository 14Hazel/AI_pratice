import langchain.embeddings
import os
from langchain.llms import OpenAI
import openai
import numpy as np
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredFileLoader
import langchain.indexes
import click


#set environment value 
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# print(os.environ['OPENAI_API_KEY'])

llm = OpenAI(temperature=0.9, batch_size=5)
text = "请解释一下sift算法"
print(llm(text))


# load Docs that we need
loader = UnstructuredHTMLLoader("test/AboutSIFT4G.html")

data =  loader.load()
print(data)

# create index
# why index because index could over token limit
index = langchain.indexes.VectorstoreIndexCreator().from_loaders([loader])

#query index

query = "请解释一下sift4g，用中文解释"
anwser = index.query_with_sources(query) #这个会返回阅读知识的来源和答案本身
query2 = "sift4g的分数判定标准是多少?"
index.query(query2)


# introduce chain so we could chose different database and keep a small talk
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff") #['stuff', 'map_reduce', 'refine', 'map_rerank']
chain2 = load_qa_chain(llm, chain_type="map_reduce")
chain.run(input_documents=data, question=query2)  #还是需要先加载文档然后做tokenize的变化，这里的文档并没有做Index的处理
chain2.run(input_documents=data, question=query)



# add search database
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.indexes.vectorstore import VectorstoreIndexCreator



with open("test/SIFT_help.html") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)
embeddings = OpenAIEmbeddings() #calculate similarity embeddings 这部分后面拆分用其他方法做


#这里就是建立数据库了
docsearch = Chroma.from_texts(texts, 
                              embeddings, metadatas=[{"source": str(i)} 
                                                    for i,v in enumerate(texts)]).as_retriever()


# 从文件中构建数据库2
loader_list = os.listdir("test/")
print(loader_list)
docs = []
loaders = [UnstructuredHTMLLoader(os.path.join("test", file)) for file in loader_list]
for loader in loaders:
    docs.extend(loader.load())

isinstance(loaders,list)
# loader = UnstructuredHTMLLoader("test/AboutSIFT4G.html")
# print(docs)
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0) #通过chunk size控制token数
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vecstore = Chroma.from_documents(documents, embeddings)


# 引入记忆机制
# introduce memory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


from langchain.chains import ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0),
                                           vecstore.as_retriever(), memory=memory)


print(query)

result = qa({"question": query})

query = "sift4G比sift快多少倍？"
result = qa({"question": query})
print(result['answer'])








# search database and found similart content
query_test = query2
docs = docsearch.get_relevant_documents(query_test)




# Add custom Prompt
from langchain.prompts import PromptTemplate


#MapReduce prompt

question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text translated into mandarin.
{context}
Question: {question}
Relevant text, if any, in Mandarin:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer chinese. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
{summaries}
=========
Answer in Mandarin:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce", 
                      return_map_steps=True, 
                      question_prompt=QUESTION_PROMPT, 
                      combine_prompt_template=COMBINE_PROMPT)



output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
print(output.get("output_text"))


# introduce chathistory so similar qa could retrieve from history
from langchain.chains import ConversationalRetrievalChain











#TODO: introduce jpeg in conversation
from langchain.document_loaders.image import UnstructuredImageLoader

def chat_pic(filepath):
    loader = UnstructuredImageLoader(filepath, mode="elements")
    data = loader.load()
    return data[0]



def doc_loader(filedir):
    if os.path.isdir(filedir):
        file_list = os.listdir(filedir)
        for file in file_list:
            if os.path.isfile(file):
                loader = UnstructuredFileLoader(file)
                return loader
            
            else:
                raise FileNotFoundError
            

#split text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len,
) 



# create index
def create_index(loader):
    index = langchain.indexes.VectorstoreIndexCreator().from_loaders([loader])
    return index


# create database
def create_db(docs):
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore



# return query base on database
def docs_qa(index, query):
    return index.query(query)



## count embedding, if you know how to count cosine_similarity 
from openai.embeddings_utils import get_embedding, cosine_similarity

text = "你好吗"
text2 = "你们好吗"
text3 = "我不好"

em1 = get_embedding(text)
em2 = get_embedding(text2)
em3 = get_embedding(text3)



if __name__=='__main__':
    pass
    # main()




#build bots with Docs




