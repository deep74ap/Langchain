from unittest import loader
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ibm.pdf")

doc = loader.load()

print(doc[9].page_content)