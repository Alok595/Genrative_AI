from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel,RunnableLambda



model=ChatMistralAI(model="mistral-tiny")
parser=StrOutputParser()



short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1-2 lines")
long_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

topic = "Machine Learning"



# formatted_Shortprompt = short_prompt.format_messages(topic=topic)

# response_short = model.invoke(formatted_Shortprompt)          now we are using runable beacuse this comment we have to call manually

# str_out = parser.parse(response_short.content)


# formatted_Longprompt = short_prompt.format_messages(topic=topic)

# response_Long = model.invoke(formatted_Longprompt)          now we are using runable beacuse this comment we have to call manually

# str_out = parser.parse(response_Long.content)


# print(final_output)



chain =RunnableParallel({
   "short":RunnableLambda(lambda x :x["short"]) | short_prompt | model | parser ,
   "detailed":RunnableLambda(lambda x :x["detailed"]) |long_prompt | model | parser


})
# result=chain.invoke(topic)



result=chain.invoke({"short":{"topic":"Machine Learning"},"detailed":{"topic":"Deep Learning"}}) #if we want to print differen topic 
print(result["short"])

print(result["detailed"])