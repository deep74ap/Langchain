from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)

#Creating a parallel chain where One chain will give the short notes for a text and one will give the quiz for that text
#Afterwards we will merge both documents

#So we need to write three prompts  : 1 for the notes, 2nd for quiz and last for the merging

prompt1 = PromptTemplate(
    template="Write a short notes in a clean and layman language on \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Provide some question and answers for the quiz for the provided text \n {text}",
    input_variables=['text']

)

prompt3 = PromptTemplate(
    template="Merge these two documents 1. notes -> {notes} and 2. Quiz -> {quiz}",
    input_variables=['notes' , 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes' : prompt1 | model | parser,
        'quiz' : prompt2 | model | parser
    }
)
merge_chain = prompt3 | model | parser


chain = parallel_chain | merge_chain


text = '''A time machine is a hypothetical or fictional device that enables time travel, allowing users to move forward or backward in time. While the concept is popular in science fiction, particularly since H.G. Wells' "The Time Machine", the existence of a real-world time machine remains speculative and unproven. Theoretical physicists have explored the idea of time travel, often involving the creation or manipulation of "closed timelike curves" in spacetime, but such concepts are far from practical realization. 
Here's a more detailed look:
Time Machines in Fiction:
Popularized by H.G. Wells:
The concept of a time machine, as a device for traversing time, was significantly popularized by H.G. Wells' 1895 novel, "The Time Machine". 
Diverse Depictions:
Time machines appear in various forms across literature, film, and other media, often with varying mechanisms and implications for time travel. 
Focus on Consequences:
Stories often explore the consequences of altering the past or future, or the paradoxes and alternate timelines that arise from time travel. 
Time Machines in Physics (Theoretical):
Closed Timelike Curves (CTCs):
In theoretical physics, time travel is sometimes linked to the idea of CTCs, where spacetime is curved in such a way that a path in spacetime can loop back on itself, allowing for a journey back to a previous point in time. 
"Weak" vs. "Strong" Time Machines:
Paul J. Nahin distinguishes between "weak" time machines (exploiting existing CTCs) and "strong" time machines (responsible for creating CTCs). 
Amos Ori's Time Donut:
Theoretical physicist Amos Ori proposed a model for a time machine using curved spacetime in a donut shape, but it relies on manipulating gravitational fields, which is beyond our current capabilities. 
Practical Considerations:
No Proven Existence:
There is no scientific evidence to support the existence of a physical time machine that allows for travel to the past or future. 
Causality Problems:
Time travel to the past raises significant questions about causality, as altering past events could lead to paradoxes or inconsistencies. 
Beyond Current Technology:
Even if theoretically possible, creating a device that manipulates spacetime to enable time travel would require technologies far beyond our current understanding and capabilities'''
res = chain.invoke({'text' : text})

print(res)


#To visualise the chain
chain.get_graph().print_ascii()