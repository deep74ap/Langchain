
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)


temp1 = PromptTemplate(template="Please write a detailed report on {topic}",
                       input_variables=['topic'])

temp2 = PromptTemplate(template="Please write the summary in only 5 line for \n {text}",
                       input_variables=['text'])

parser = StrOutputParser()


chain = temp1 | model | parser | temp2 | model  |parser

res = chain.invoke({'topic' : 'Cricket'})

print(res)
chain.get_graph().print_ascii()