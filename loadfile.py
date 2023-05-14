from langchain.document_loaders import UnstructuredHTMLLoader
import os, os.path



# avoid ssl error


loader = UnstructuredHTMLLoader("test/AboutSIFT4G.html")

data =  loader.load()
print(data)



# TODO
"""
用class对文档内容加载进行进一步的封装
"""

class Docs:
    def __init__(self, filepath:str, filetype:str) -> str:

        self.filepath = filepath

        self.filetype = filetype

    def type(self):
        #按照文件后缀名切割？
        pass

    # def check(self):

    #     if 



