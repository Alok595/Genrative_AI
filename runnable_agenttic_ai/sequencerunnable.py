from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")


model=ChatMistralAI(model="mistral-tiny")

parser=StrOutputParser()

# formatted_prompt = prompt.format_messages(topic="Machine Learning")

# response = model.invoke(formatted_prompt)          now we are using runable beacuse this comment we have to call manually

# final_output = parser.parse(response.content)

# print(final_output)



chain = prompt | model | parser


result=chain.invoke("Deep Learnings")
print(result)