from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough





model=ChatMistralAI(model="mistral-tiny")
parser=StrOutputParser()


code_prompt = ChatPromptTemplate.from_messages([
  ("system","You are a code genrater"),
  ("human","{topic}")
])


explain_prompt = ChatPromptTemplate.from_messages([
  ("system","You are a helpful  assistant who explain code in simple terms"),("human","Explain the following code in simple words:\n{code}")
])

# seq = code_prompt | model | parser | explain_prompt | model | parser  it has a issue like it dont give code beacue it explain the code and give output as explain
 #to solve this 

seq = code_prompt | model | parser 


seq2 = RunnableParallel({"code":RunnablePassthrough(),
                         "explation":explain_prompt | model |parser})


chain = seq | seq2

result = chain.invoke({"topic":"Write a code of plaindrop in python"})

print(result['code'])
print(result['explation'])





