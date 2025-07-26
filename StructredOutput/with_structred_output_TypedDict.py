from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated , Optional,Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")

#Creating Schema for the output

class Feedback(TypedDict):

    #Annotated helps to define what should be the output llm need to give in this section
    summary: Annotated[str,"A brief summary for the provided feedback"]

    #Literal helps to categorize the sentiments based on  the feeddback
    sentiment: Annotated[Literal["pos" , "neg" , "neutral"] , "Return the sentiment of review"]

    #Optional helps LLM to decide whether give output for this section or not based on data
    
    Rating : Annotated[Optional[int] , "Rating of the car based on feedback within range of 1 to 5"]

#Creating structred output model

structred_Model = model.with_structured_output(Feedback)

res  = structred_Model.invoke('''The Tesla Model Y is a stylish, all-electric SUV known for its impressive range, fast acceleration, and advanced tech features. It offers a spacious interior, strong safety ratings, and benefits from Tesla’s Supercharger network. Owners appreciate the minimalist design and over-the-air updates. However, the ride can be firm, and build quality is inconsistent, with some reporting rattles or panel gaps. Reliability ratings are average, and the infotainment system lacks physical controls, which some find frustrating. Overall, the Model Y delivers great performance and value for an EV, but may fall short on luxury and refinement compared to premium rivals.''')

print(res)