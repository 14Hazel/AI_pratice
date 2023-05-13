from langchain.document_loaders import UnstructuredHTMLLoader
import nltk
import ssl



# avoid ssl error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')


loader = UnstructuredHTMLLoader("test/AboutSIFT4G.html")

data =  loader.load()
print(data)



# TODO
"""
用class对文档内容加载进行进一步的封装
"""

class Docs:
    def __init__() -> None:
        pass
