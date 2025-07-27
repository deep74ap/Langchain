from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= 'text-generation'
)
model = ChatHuggingFace(llm = llm)

temp1 = PromptTemplate(template="Please write a detailed report on {topic}",
                       input_variables=['topic'])

temp2 = PromptTemplate(template="Please write the summary in only 5 line for /n {text}",
                       input_variables=['text'])

parser = StrOutputParser()


chain = temp1 | model | parser | temp2 | model  |parser

res = chain.invoke({'topic' : 'Cricket'})

print(res)