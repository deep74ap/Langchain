from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")
#Chat History helps in maintaining the memory
chatHistory = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:
    userInput = input('You : ')
    chatHistory.append(HumanMessage(content=userInput))  #Appending messages to chat history with human message labeled
    if userInput == 'exit':
        break
    res = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=res.content))  ##Appending AI result messages to chat history with AI message labeled
    print("AI : ",res.content)

print(chatHistory)