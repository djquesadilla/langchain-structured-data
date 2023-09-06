from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()  # take environment variables from .env so we can load OPENAI_API_KEY

person_json_schema = {
    "title": "Person",
    "description": "Identifying information about a person.",
    "type": "object",
    "properties": {
        "name": {"title": "Name", "description": "The person's name", "type": "string"},
        "age": {"title": "Age", "description": "The person's age", "type": "integer"},
        "fav_food": {
            "title": "Fav Food",
            "description": "The person's favorite food",
            "type": "string",
        },
    },
    "required": ["name", "age"],
}

people_json_schema = {
    "title": "People",
    "description": "A list of people.",
    "type": "object",
    "properties": {
        "people": {
            "title": "People",
            "description": "A list of people",
            "type": "array",
            "items": person_json_schema
        }
    }
}

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting information in structured formats."),
        ("human", "Use the given format to generate {number} random person with {type} names.")
    ]
)

chain = create_structured_output_chain(people_json_schema, llm, prompt, verbose=True)
output = chain.run({
    "number": 5,
    "type": "american"
})

print(output)
