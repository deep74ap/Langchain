from langchain_core.prompts import PromptTemplate

cricket_prompt = PromptTemplate(
    template = """
    You are a cricket statistician. Given the cricketer "{player}", format "{format}", and venue type "{venue}", 
    provide the following:

    - Total matches played
    - Total runs scored
    - Batting average
    - Any notable high scores

    Respond in a clean, readable format. If the player hasn't played in that condition, say "Data not available".
    """,
    input_variables = ['player' , 'format' , 'venue'],
    validate_template = True)


cricket_prompt.save("template.json")