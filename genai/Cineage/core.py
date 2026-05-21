from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


from langchain.chat_models import init_chat_model


model = init_chat_model("mistral-small-2603")

# from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert movie information extraction assistant.

Your task is to analyze a movie-related paragraph and extract the most useful structured information.

Present the information in a clear and well-organized format using headings and bullet points.

Include the following sections:
- Movie Name
- Director
- Main Cast
- Characters
- Genre
- Setting
- Plot Summary
- Themes
- Key Messages
- Famous Dialogues
- Character Goals
- Conflicts
- Emotional Tone
- Target Audience
- Quick Summary
- One Line Summary

Rules:
- Infer information when clearly implied.
- If information is missing, write "Not Available".
- Do not return JSON.
- Do not use markdown code blocks.
- Format the output in a clean and readable way.
        """
    ),
    (
        "human",
        """
Analyze the following movie paragraph and extract the required structured information.

Paragraph:
{paragraph}
        """
    )
])


para= input("Give Your Paragraph:")

final_prompt= prompt.invoke({
"paragraph":para
})



response = model.invoke(final_prompt)
  
print(response.content)


