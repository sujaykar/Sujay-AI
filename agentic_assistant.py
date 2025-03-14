#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

class AgenticAssistant:
    def __init__(self, vector_db, model_name="gpt-3.5-turbo", temperature=0):
        self.vector_db = vector_db
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize the agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize the tools available to the agent."""
        tools = [
            Tool(
                name="Document Search",
                func=self.search_documents,
                description="Useful for searching through documents. Input should be a query string."
            ),
            Tool(
                name="Document Analysis",
                func=self.analyze_documents,
                description="Useful for analyzing documents to extract insights. Input should be a query about the documents."
            ),
            Tool(
                name="Data Visualization",
                func=self.visualize_data,
                description="Useful for creating visualizations from data. Input should be instructions for creating a visualization."
            )
        ]
        return tools
    
    def search_documents(self, query: str, collection_name: str = "default", k: int = 5) -> str:
        """Search for documents relevant to the query."""
        results = self.vector_db.search(query, k=k, collection_name=collection_name)
        
        if not results:
            return "No relevant documents found."
        
        context = "\n\n".join([doc.page_content for doc in results])
        return f"Here are the relevant documents:\n\n{context}"
    
    def analyze_documents(self, query: str, collection_name: str = "default") -> str:
        """Analyze documents to extract insights."""
        # Create a retrieval QA chain
        retriever = self.vector_db.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        qa_prompt_template = """
        You are an expert data analyst. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a detailed analysis with insights and recommendations:
        """
        
        qa_prompt = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        result = qa_chain.run(query)
        return result
    
    def visualize_data(self, instructions: str) -> str:
        """Create visualizations based on the instructions."""
        # This is a placeholder. In a real application, you would parse the instructions
        # and create appropriate visualizations.
        return "Visualization feature is not yet implemented."
    
    def run(self, query: str) -> str:
        """Run the agent with the given query."""
        try:
            response = self.agent.run(input=query)
            return response
        except Exception as e:
            return f"Error: {str(e)}"


# In[ ]:




