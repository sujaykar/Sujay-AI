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
        """Initialize the AI-powered assistant."""
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
        5. "
