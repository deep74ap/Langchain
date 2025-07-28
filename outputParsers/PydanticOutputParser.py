

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Optional,Literal
from pydantic import BaseModel , Field
from regex import template

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)



class ReviewAnalysis(BaseModel):
    summary: str
    sentiment: str = Field(description="Sentiment: Positive, Negative, or Neutral")
    rating: float = Field(description="Rating out of 5")

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

prompt = PromptTemplate(
    template=(
        "Analyze the following review text and return:\n"
        "- A brief summary\n"
        "- An emotional tone (sentiment)\n"
        "- A rating out of 5\n\n"
        "Respond ONLY with JSON.\n\n"
        "{format_instructions}\n\n"
        "Review:\n{text}"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

text = '''The Tesla Model Y is a stylish, all-electric SUV known for its impressive range, fast acceleration, and advanced tech features. It offers a spacious interior, strong safety ratings, and benefits from Tesla’s Supercharger network. Owners appreciate the minimalist design and over-the-air updates. However, the ride can be firm, and build quality is inconsistent, with some reporting rattles or panel gaps. Reliability ratings are average, and the infotainment system lacks physical controls, which some find frustrating. Overall, the Model Y delivers great performance and value for an EV, but may fall short on luxury and refinement compared to premium rivals.'''

chain = prompt | model | parser

res = chain.invoke({'text' : text})
print(res)