from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= 'text-generation'
)

model = ChatHuggingFace(llm = llm)

res = model.invoke("Who is Virat kohli")

print(res)