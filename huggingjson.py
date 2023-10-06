from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFace
from langchain.chains import LLMChain
import json

# Configura tu clave de API de OpenAI
api_key = ""

# Lee los datos desde el archivo JSON
with open("ejemplo.json", "r") as archivo_json:
    datos_json = json.load(archivo_json)

# Extrae los datos correspondientes al campo "text"
texto=datos_json['textAnnotations'][0]['description']
#datos_texto = datos_json.get("textAnnotations", [])
print(texto)
prompt=PromptTemplate(
    input_variables=["text"],
    template="Extrae el nombre y la edad del siguiente texto:\n\n\"{text}\"",
)
llm=HuggingFace(
    #api_key=api_key,
    model_name="gpt-neo-2.7B",
    temperature=0.9
)
llmchain=LLMChain(llm=llm, prompt=prompt)
llmchain.run(texto)
