# from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv


load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task= 'text-generation'
# )
# model = ChatHuggingFace(llm = llm)


parser = JsonOutputParser()

template = PromptTemplate(
     template=(
        "Give me a name of a person , age and city of a country {country} \n {format_instruction}"
    ),
    input_variables=['country'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}

)
chain  = template | model | parser

res = chain.invoke({'country' : 'India'})
print(res)