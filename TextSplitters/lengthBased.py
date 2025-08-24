from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from torch import chunk

file_path = "ibm.pdf"
loader = PyPDFLoader(file_path)
doc = loader.load();
# print(doc[6].page_content)
splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 10,   #Overlap between characters
    separator = ''
)

# text = "You are a cricket statistician. Given the cricketer , format , and venue type , provide the following: Total matches playedTotal runs scoredBatting averageAny notable high scoresRespond in a clean, readable format. If the player hasn't played in that condition, say "
# res = splitter.split_text(text);



res = splitter.split_documents(doc)
print(res[6].page_content)