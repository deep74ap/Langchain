from pydoc import describe
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from typing import Literal

import pydantic


load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)

parser = StrOutputParser()


class FeeBack(BaseModel):
    sentiment : Literal['Positive' , 'Negative'] = Field(description="What is overall tone of the text")

pydanticParser = PydanticOutputParser(pydantic_object=FeeBack)

prompt1 = PromptTemplate(
    template="You have to classify the overall emotional tone or sentiment for the provided text /n {text} /n {format_instruction}",
    input_variables=['text'],
    partial_variables={'format_instruction': pydanticParser.get_format_instructions()}
    

)

classifier_chain = prompt1 | model | pydanticParser

prompt2 = PromptTemplate(
    template="Write an appropriate response for {tone} feedback",
    input_variables=['tone']
)
prompt3 = PromptTemplate(
    template="Write an appropriate response for {tone} feedback",
    input_variables=['tone']
)

branch = RunnableBranch(
    (lambda x : x.sentiment == 'Positive' , prompt2 | model | parser),
    (lambda x: x.sentiment == 'Negative' , prompt3 | model |parser),
    RunnableLambda(lambda x : "could not find sentiment")
)

chain = classifier_chain | branch

res = chain.invoke("This is a terrible car")
print(res)


chain.get_graph().print_ascii()