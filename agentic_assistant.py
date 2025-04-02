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
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
import requests
from io import BytesIO

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
                name="Career Coach",
                func=self.career_coach,
                description="Analyzes job descriptions and resumes, identifies skill gaps, and provides interview preparation guidance."
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
            ),
            Tool(
            name="PowerPoint Generator with DALL·E",
            func=self.generate_presentation,
            description="Creates a PowerPoint presentation with AI-generated images and text based on the user's query or relevant database content."
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
        results = self.vector_db.search(query, k=10, collection_name=None)

        if not results or results[0].page_content == "No relevant documents found.":
            return "No relevant course materials found."

        context = "\n\n".join([doc.page_content for doc in results])

        prompt_template = """
        You are a highly qualified STEM professor at a top-tier university.
        Generate 30-50 practice questions across different difficulty levels (Easy, Medium, Hard) based on the course material.

        Course Material:
        {context}

        Practice Questions:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        questions = self.llm.predict(prompt.format(context=context))
        return questions

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

    def career_coach(self, query: str) -> str:
        """Analyzes job descriptions and resumes, identifies skill gaps, and provides interview preparation."""
        results = self.vector_db.search(query, k=5)
        if not results:
            return "No relevant job descriptions or resumes found."
        
        context = "\n\n".join([doc.page_content for doc in results])
        
        prompt_template = """
        You are an expert Career Coach. Based on the following job description and resume, analyze the candidate's fit, highlight skill gaps, and provide interview preparation guidance.

        Job Description and Resume:
        {context}

        Analysis and Recommendations:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        response = self.llm.predict(prompt.format(context=context))
        return response
        

        def generate_presentation(self, query: str, num_slides: int = 5, use_vector_db: bool = False) -> str:
        """Generates a PowerPoint presentation based on a user query or vector database content."""
        
        # Retrieve relevant content from vector database if specified
        if use_vector_db:
            results = self.vector_db.search(query, k=5)
            if not results or results[0].page_content == "No relevant documents found.":
                return "No relevant documents found in the vector database."
            context = "\n\n".join([doc.page_content for doc in results])
        else:
            context = query  # Use user-provided content if no vector search

        # Define structured prompt
        prompt_template = """
        You are an expert presentation designer. Create a structured PowerPoint presentation outline.
        Ensure the outline contains {num_slides} slides with:
        - Slide titles
        - Bullet points summarizing key ideas
        - Suggestions for visuals
        
        Content Reference:
        {context}
        
        Provide output in the following format:
        Slide 1 Title: <title>
        - <bullet point 1>
        - <bullet point 2>
        
        Slide 2 Title: <title>
        - <bullet point 1>
        - <bullet point 2>
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "num_slides"])
        response = self.llm.predict(prompt.format(context=context, num_slides=num_slides))
        
        # Parse response into slides
        slides = self._parse_slides(response)
        
        # Create PowerPoint file
        ppt_path = self._create_pptx(query, slides)
        
        return ppt_path
    
    def _parse_slides(self, response: str) -> List[Dict[str, List[str]]]:
        """Parses the AI-generated response into a structured slide format."""
        slides = []
        lines = response.split("\n")
        current_slide = None
        
        for line in lines:
            if line.startswith("Slide"):
                if current_slide:
                    slides.append(current_slide)
                current_slide = {"title": line.split(": ")[1], "content": []}
            elif line.startswith("-"):
                if current_slide:
                    current_slide["content"].append(line.strip("- "))
        
        if current_slide:
            slides.append(current_slide)
        
        return slides
    
    def _create_pptx(self, topic: str, slides: List[Dict[str, List[str]]]) -> str:
        """Creates a PowerPoint file from the AI-generated slides."""
        prs = Presentation()
        
        for slide_data in slides:
            slide_layout = prs.slide_layouts[1]  # Title and Content layout
            slide = prs.slides.add_slide(slide_layout)
            title = slide.shapes.title
            content = slide.placeholders[1]
            
            title.text = slide_data["title"]
            content.text = "\n".join(slide_data["content"])
            
            # Generate an image for the slide using DALL·E
            image_path = self._generate_slide_image(slide_data["title"])
            if image_path:
                left = Inches(5)
                top = Inches(2)
                slide.shapes.add_picture(image_path, left, top, width=Inches(4))
        
        ppt_path = f"{topic.replace(' ', '_')}.pptx"
        prs.save(ppt_path)
        return ppt_path
    
    def _generate_slide_image(self, description: str) -> Optional[str]:
        """Generates an image using DALL·E based on the slide title."""
        dalle_url = "https://api.openai.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": "dall-e-3", "prompt": description, "n": 1, "size": "1024x1024"}
        
        response = requests.post(dalle_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            image_url = response.json()["data"][0]["url"]
            img_response = requests.get(image_url)
            img = Image.open(BytesIO(img_response.content))
            
            image_path = f"{description.replace(' ', '_')}.png"
            img.save(image_path)
            return image_path
        
        return None

