from langchain_openai import ChatOpenAI
from typing import List, Dict, Tuple, Any
import time
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

class AdvancedRAGPipeline:
    def __init__(self, retriever, model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=1042):
        self.retriever=retriever
        self.llm=ChatOpenAI(api_key=openai_api_key,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            model_name=model_name
                            )
        self.history=[]

    def query(self, question: str, top_k: int=5, min_score: float= 0.2, stream: bool= False, summarize: bool= False)-> Dict[str,Any]:
        # Retrieved Relevent documents
        results=self.retriever.retrieve(question, top_k = top_k, score_threshold=min_score)
        if not results:
            answer = "No relevent context found"
            sources=[]
            context=""
        else:
            # Build context from retrieved documents
            context="\n\n".join([doc['content'] for doc in results])
            sources=[{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source','unknown')),
                'page': doc['metadata'].get('page','unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...' 
            }for doc in results]
            # Streaming answer similuation
            prompt=f"""Use the following context to answer the question concisely.\nContext: \n{context}\n\nQuestion:{question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content
        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer
        # Optionally summerize the answer
        summary=None
        if summarize and answer:
            summary_prompt=f"Summerize the following answer in 2 lines : \n{answer}"
            summary_resp=self.llm.invoke([summary_prompt])
            summary=summary_resp.content
        
        # Store query history
        self.history.append({
            "question":question,
            'answer':answer,
            'sources':sources,
            'summary':summary
        })

        return{
            'question':question,
            'answer':answer_with_citations,
            'sources':sources,
            'summary':summary,
            'history':self.history
        }