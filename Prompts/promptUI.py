from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()

# st.header("Research Assistant Tool")

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro",temperature=0.7)

# user_inp = st.text_input("Please enter the prompt")

# Streamlit App Title
st.title("🏏 Cricket Stats Finder")


cricketers = ["Virat Kohli", "Rohit Sharma", "MS Dhoni", "Steve Smith", "Kane Williamson"]
formats = ["Test", "ODI", "T20"]
venues = ["Home", "Away"]

selected_player = st.selectbox("Select Cricketer", cricketers)
selected_format = st.selectbox("Select Format", formats)
selected_venue = st.selectbox("Select Venue", venues)

st.markdown(f"### You selected:")
st.write(f"**Cricketer:** {selected_player}")
st.write(f"**Format:** {selected_format}")
st.write(f"**Venue:** {selected_venue}")

template = load_prompt('template.json')





if st.button("Send"):
    chain = template | model

    res = chain.invoke({
        'player' : selected_player,
        'format' : selected_format,
        'venue' : selected_venue
    })
    st.write(res.content)