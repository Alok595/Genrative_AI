from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()

from langchain.chat_models import init_chat_model



model = init_chat_model("mistral-small-2603")


class Movie(BaseModel):
  title:str
  release_year:Optional[int]
  genre:List[str]
  director:Optional[str]
  cast:List[str]
  rating:Optional[float]
  summary:str


parser=PydanticOutputParser(pydantic_object=Movie)




# from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system',""" 

Extract movie information from the paragraph
{format_intructions}


"""),("human","{paragraph}")
])


para= input("Give Your Paragraph:")

final_prompt= prompt.invoke({
"paragraph":para,
"format_intructions":parser.get_format_instructions()
})



response = model.invoke(final_prompt)
  
print(response.content)


