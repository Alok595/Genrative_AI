from dotenv import load_dotenv

# Loads environment variables from a .env file into the system environment
load_dotenv()



from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
model = init_chat_model("mistral-small-2603")


print("Choose Your Ai Agent Mode")
print("press 1 for Angry Mode")
print("press 2 for Funny Mode")
print("press 3 for Sad Mode")

choice = int(input("Enter Your Choise:-"))

if choice == 1:
  mode= "You are an angry Ai Agent. You respond aggresively and imapatiently"
elif choice ==2:
   mode= "You are an funny Ai Agent. You respond humor and jokes"
elif choice ==3:
   mode= "You are an sad Ai Agent. You respond sad and aggresively"


messages = [

SystemMessage(content=mode)


]   # it is short term meomry, there is a issue with this like stoarge so we can use message sin langchan to storte history

print("--------Welcome Type 0 for exit")
while True:

  
  prompt = input("You:")
  # messages.append(prompt)
  messages.append(HumanMessage(content=prompt))
  if prompt=="0":
    break


  # response = model.invoke(prompt)  #not storing the history
  response = model.invoke(messages)  
  # messages.append(response.content) 
  messages.append(AIMessage(content=response.content))
  print("Bot:",response.content)



print(messages)
