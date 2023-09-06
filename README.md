# Langchain Structured Data

This repository aims to demonstrate how to incorporate OpenAI function-calling API's in a Langchain chain to output structured data.

## Project setup

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Description

Working with LLMs is fun but sometimes too much free text is a pain to parse and process.

Fortunately, with Langchain and OpenAI functions we can structure data in a few lines of code.

Before OpenAI functions I was prompting stuff like “Please, give me the output in the following JSON structure” and praying to the gods that it would magically solve my problems.

Unfortunately I was being given replies with structured data prefixed with “Sure! Here’s your data:” or suffixed with “Hope this structure works for you. Enjoy!”.

## Getting structured outputs of given information

We can use **JsonSchema** (Pydantic is awesome but it is still very buggy 🐛 for Langchain v0.0.281 at the time I wrote this article) to structure whatever we want. Let’s use the straightforward example from [Langchain documentation](https://python.langchain.com/docs/modules/chains/how_to/openai_functions) to parse someone’s age:

```python
from typing import Optional

from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()  # take environment variables from .env so we can load OPENAI_API_KEY

json_schema = {
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

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}")
    ]
)

chain = create_structured_output_chain(json_schema, llm, prompt, verbose=True)
output = chain.run("DJ Quesadilla is 38 and loves pizza")

print(output)
```

## **Getting structured outputs of generated information**

Sometimes we need to generate data on the fly and output it in the right format. Let’s take the previous example and generate a random list of *5 people with american names* and their favourite food.

We just need to add a new `people_json_schema` to create an array of people and pass it to the `create_structured_output_chain`:

```python
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
```

