#Dynamic prompt for the list of messages

from langchain_core.prompts import ChatPromptTemplate

chat_Temp = ChatPromptTemplate.from_messages([
    ("system", "You are expert in {domain}"),
    ("human", "Tell me about {topic}")
])

prompt = chat_Temp.invoke({'domain' : 'cricket' ,
                  'topic' : 'Hitwicket'})

print(prompt)