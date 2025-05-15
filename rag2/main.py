from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")
template = """
You are an expert in infering the sentiment in US at a given time based on social media posts .

Here are some relevant posts : {posts}

Here is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    soc_posts = retriever.invoke(question)
    result = chain.invoke({"posts": soc_posts, "question": question})
    print(result)
