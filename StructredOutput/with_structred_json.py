from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)


# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""The Tesla Model Y is a versatile electric SUV that embodies Tesla’s commitment to innovation, sustainability, and futuristic design. Built on the same platform as the Model 3, it offers a spacious interior with an optional third-row seat, making it a practical choice for families or individuals needing extra cargo room. One of its standout themes is technology, evident through its minimalist dashboard dominated by a central touchscreen, over-the-air software updates, and advanced driver-assistance systems like Autopilot. In terms of performance, the Model Y delivers impressive acceleration and range, especially in the Long Range variant, which can exceed 500 km on a single charge under ideal conditions. However, the car is not without its flaws. Some users have reported inconsistent build quality, including panel misalignments and interior trim issues, which detract from the premium experience expected at its price point. Additionally, the suspension is on the firmer side, making the ride feel less smooth over uneven roads. The Full Self-Driving (FSD) package is promising but still in beta and comes at a high cost. Overall, the Tesla Model Y earns a strong 4.2 out of 5, offering excellent electric performance and advanced tech, though it still has room to improve in comfort and finish.
                                 
Review by Deepak
""")

print(result)