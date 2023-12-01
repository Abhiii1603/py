import os
from secret_key import openapi_key
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


os.environ["OPENAI_API_KEY"] = openapi_key


llm = OpenAI(temperature=1)


def generate_restaurant_name_and_items(cuisine):
    prompt_temp_name = PromptTemplate(
    input_variables= ["cuisine"], 
    template= "I want to open a restaurant for {cuisine} food. Suggest a fancy name for it."
)

    name_chain = LLMChain(llm = llm, prompt= prompt_temp_name, output_key="restaurant_name")

    prompt_temp_items = PromptTemplate(
        input_variables=["restaurant_name"], 
        template = "Suggest some menu items for {restaurant_name}, Return it as csv.")


    food_chain_items = LLMChain(llm = llm, prompt=prompt_temp_items, output_key="menu_items")




    chain = SequentialChain(chains = [name_chain, food_chain_items], 
                        input_variables= ["cuisine", ], 
                        output_variables = ["restaurant_name", "menu_items"])

    response = chain({"cuisine":"India"})
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))
    




