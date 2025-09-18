from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

class RagPipeline:
    def __init__(self, retriever, model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=1042):
        self.retriever=retriever
        self.llm=ChatOpenAI(api_key=openai_api_key,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            model_name=model_name
                            )
        
    def query(self, question, top_k=3):
        # Retrieved context
        results=self.retriever.retrieve(question, top_k=top_k)
        context="\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return "No relevent context found"
        # Prompt
        prompt=f"""Use the following content to answer the question concisely
        context:{context}

        Question: {question}

        
        Answer:"""
        response=self.llm.invoke([prompt.format(context=context, question=question)])
        return response.content
        