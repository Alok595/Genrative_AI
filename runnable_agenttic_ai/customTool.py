from langchain.tools import tool

@tool
def get_gretting(name:str)->str:
  '''
  Genrate a gretting message for a user
  '''
  return f"Hellow {name} , Wlcome to here"

result=get_gretting.invoke({"name":"alok"})
print(result)