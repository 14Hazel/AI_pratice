import langchain.embeddings
import os
from langchain.llms import OpenAI
import openai
import numpy as np
from langchain.document_loaders import UnstructuredHTMLLoader
import langchain.indexes


#set environment value 
os.environ['OPENAI_API_KEY'] = "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"

# print(os.environ['OPENAI_API_KEY'])

llm = OpenAI(temperature=0.9)
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
query = "请解释一下sift4g"
index.query(query)
query2 = "sift4g的分数判定标准是多少?"
index.query(query2)



#count embedding, if you know how to count cosine_similarity
from openai.embeddings_utils import get_embedding, cosine_similarity

text = "你好吗"
text2 = "你们好吗"
text3 = "我不好"

em1 = get_embedding(text)
em2 = get_embedding(text2)
em3 = get_embedding(text3)





#



#build bots with Docs




