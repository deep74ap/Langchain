from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)

res = model.invoke("Write  a five line poem on love")
print(res.content)