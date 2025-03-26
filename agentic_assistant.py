#!/usr/bin/env python
# coding: utf-8

import os
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

class AgenticAssistant:
    def __init__(self, vector_db, model_name="o3-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY")):
        self.vector_db = vector_db
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

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
                name="Document Summarization",
                func=self.summarize_document,
                description="Summarizes key points from a document. Input should be a query about a document."
            ),
            Tool(
                name="Chart Analysis",
                func=self.analyze_charts,
                description="Analyzes charts or graphs and extracts insights. Input should be a query about a chart or dataset."
            ),
            Tool(
                name="Financial Data Insights",
                func=self.analyze_financial_data,
                description="Analyzes financial reports, stock trends, and market insights."
            ),
            Tool(
                name="General Question Answering",
                func=self.answer_general_query,
                description="Handles general queries using document search."
            ),
            Tool(
                name="STEM Professor - Practice Questions",
                func=self.generate_practice_questions,
                description="Reads course material and generates practice questions for students to test their knowledge."
            )
        ]
        return tools

    def classify_query(self, query: str) -> str:
        """Classifies the user's query to determine the relevant expert agent."""
        classification_prompt = f"""
        You are an AI query classifier. Categorize the following user query into one of these categories:

        1. "document_summarization" - if the query is about summarizing a document.
        2. "chart_analysis" - if the query is about analyzing a chart or visualization.
        3. "financial_analysis" - if the query is about financial reports, stock trends, or market insights.
        4. "general_qa" - if the query is a general question.
        5. "practice_questions" - if the query asks for practice questions on a topic.

        Query: "{query}"
        Output ONLY the category as a string.
        """
        
        response = self.llm.predict(classification_prompt)
        return response.strip().lower()

    def run(self, query: str) -> str:
        """Routes the query to the appropriate expert agent based on classification."""
        try:
            category = self.classify_query(query)

            if category == "document_summarization":
                return self.summarize_document(query)
            elif category == "chart_analysis":
                return self.analyze_charts(query)
            elif category == "financial_analysis":
                return self.analyze_financial_data(query)
            elif category == "general_qa":
                return self.answer_general_query(query)
            elif category == "practice_questions":
                return self.generate_practice_questions(query)
            else:
                return "I'm not sure how to categorize your request. Can you clarify?"

        except Exception as e:
            return f"Error: {str(e)}"

    def generate_practice_questions(self, query: str) -> str:
        """Generates 30-50 practice questions based on a specific topic, module, or chapter from course material."""
        
        # ✅ Ensure search includes all collections
        results = self.vector_db.search(query, k=10, collection_name=None)

        if not results or results[0].page_content == "No relevant documents found.":
            return "No relevant course materials found."

        context = "\n\n".join([doc.page_content for doc in results])

        prompt_template = """
        You are a highly qualified STEM professor at a premium college.  
        Based on the following course material, generate a set of **30-50 practice questions** to test student knowledge.
        
        The questions should cover different levels of difficulty:
        - **Easy:** Basic concept recall.
        - **Medium:** Application-based problems.
        - **Hard:** Advanced problem-solving.

        Additionally, provide a **separate answer key** with detailed explanations.

        **Course Material:**
        {context}

        **Practice Questions:**
        """

        # Generate questions
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        questions = self.llm.predict(prompt.format(context=context))

        answer_prompt_template = """
        You are a highly qualified STEM professor at a premium college.  
        Based on the **practice questions** below, generate a **detailed answer key** for each question with explanations.

        **Practice Questions:**
        {questions}

        **Answer Key:**
        """
        answer_prompt = PromptTemplate(template=answer_prompt_template, input_variables=["questions"])
        answers = self.llm.predict(answer_prompt.format(questions=questions))

        return f"### Practice Questions:\n{questions}\n\n### Answer Key:\n{answers}"
    
    def analyze_charts(self, query: str) -> str:
        """Analyzes charts and provides insights."""
        results = self.vector_db.search(query, k=5)
        if not results:
            return "No relevant charts found."
        
        context = "\n\n".join([doc.page_content for doc in results])

        prompt_template = """
        You are a data analyst skilled in visualizations. Analyze the following charts and provide key insights.

        Charts:
        {context}

        Insights:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        response = self.llm.predict(prompt.format(context=context))
        return response

    def analyze_financial_data(self, query: str) -> str:
        """Analyzes financial reports, stock market trends, and business data."""
        results = self.vector_db.search(query, k=5)
        if not results:
            return "No relevant financial data found."
        
        context = "\n\n".join([doc.page_content for doc in results])

        prompt_template = """
        You are an expert financial analyst. Review the following financial data and extract key insights.

        Financial Data:
        {context}

        Analysis and Recommendations:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        response = self.llm.predict(prompt.format(context=context))
        return response

    def answer_general_query(self, query: str) -> str:
        """Handles general queries by searching documents and providing answers."""
        results = self.vector_db.search(query, k=5)
        if not results:
            return "I couldn't find any relevant information."

        context = "\n\n".join([doc.page_content for doc in results])
        prompt_template = """
        Use the provided context to answer the following question.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        response = self.llm.predict(prompt.format(context=context, query=query))
        return response

    def summarize_document(self, query: str) -> str:
        """Summarizes the document content based on the user's query."""
        results = self.vector_db.search(query, k=5)
        if not results:
            return "No relevant documents found."
        
        context = "\n\n".join([doc.page_content for doc in results])

        prompt_template = """
        You are an expert document summarizer. Read the following document and summarize the key points concisely.

        Document:
        {context}

        Summary:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        response = self.llm.predict(prompt.format(context=context))
        return response
