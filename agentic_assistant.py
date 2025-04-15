#!/usr/bin/env python
# coding: utf-8

import os
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.agents import Tool
from langchain.chains import RetrievalQA
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
from image_creator import DalleImageGenerator
import re
from langchain import hub
from langchain.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Define the state for the Langgraph
@dataclass
class AgentState:
    query: str
    chat_history: List[BaseMessage] = field(default_factory=list)
    retrieved_context: Optional[List[Document]] = None
    intermediate_output: Optional[Union[str, dict]] = None
    image_path: Optional[str] = None
    ppt_path: Optional[str] = None

# Initialize LLM
llm = ChatOpenAI(model_name="chatgpt-4o-latest", temperature=0.7)
image_generator = DalleImageGenerator()
vector_db_instance = None # Will be set in __init__

# Define tools as Langchain tools
@tool
def summarize_document(query: str) -> str:
    """Summarizes key points from a document. Input should be a query about a document."""
    results = vector_db_instance.search(query, k=5)
    if not results:
        return "No relevant documents found."
    context = "\n\n".join([doc.page_content for doc in results])
    prompt_template = """You are an expert document summarizer. Read the following document and summarize the key points concisely.\n\nDocument:\n{context}\n\nSummary:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return llm.predict(prompt.format(context=context))

@tool
def career_coach(query: str) -> str:
    """Analyzes job descriptions and resumes, identifies skill gaps, and provides interview preparation guidance."""
    results = vector_db_instance.search(query, k=5)
    if not results:
        return "No relevant job descriptions or resumes found."
    context = "\n\n".join([doc.page_content for doc in results])
    prompt_template = """You are an expert Career Coach. Based on the following job description and resume, analyze the candidate's fit, highlight skill gaps, and provide interview preparation guidance.\n\nJob Description and Resume:\n{context}\n\nAnalysis and Recommendations:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return llm.predict(prompt.format(context=context))

@tool
def analyze_charts(query: str) -> str:
    """Analyzes charts or graphs and extracts insights. Input should be a query about a chart or dataset."""
    results = vector_db_instance.search(query, k=5)
    if not results:
        return "No relevant charts found."
    context = "\n\n".join([doc.page_content for doc in results])
    prompt_template = """You are a data analyst skilled in visualizations. Analyze the following charts and provide key insights.\n\nCharts:\n{context}\n\nInsights:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return llm.predict(prompt.format(context=context))

@tool
def analyze_financial_data(query: str) -> str:
    """Analyzes financial reports, stock trends, and market insights."""
    results = vector_db_instance.search(query, k=5)
    if not results:
        return "No relevant financial data found."
    context = "\n\n".join([doc.page_content for doc in results])
    prompt_template = """You are an expert financial analyst. Review the following financial data and extract key insights.\n\nFinancial Data:\n{context}\n\nAnalysis and Recommendations:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return llm.predict(prompt.format(context=context))

@tool
def answer_general_query(query: str) -> str:
    """Handles general queries using document search."""
    results = vector_db_instance.search(query, k=5)
    if not results:
        return "I couldn't find any relevant information."
    context = "\n\n".join([doc.page_content for doc in results])
    prompt_template = """Use the provided context to answer the following question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    return llm.predict(prompt.format(context=context, query=query))

@tool
def generate_practice_questions(query: str) -> str:
    """Generates 30-50 practice questions based on course material."""
    results = vector_db_instance.search(query, k=10, collection_name=None)
    if not results or results[0].page_content == "No relevant documents found.":
        return "No relevant course materials found."
    context = "\n\n".join([doc.page_content for doc in results])
    prompt_template = """You are a highly qualified STEM professor. Generate 30-50 practice questions across different difficulty levels (Easy, Medium, Hard) based on the course material.\n\nCourse Material:\n{context}\n\nPractice Questions:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return llm.predict(prompt.format(context=context))

@tool
def create_single_image(prompt: str) -> str:
    """Generates a single image using DalleImageGenerator based on the prompt."""
    print(f"Tool 'Image Creator' called with prompt: {prompt[:100]}...")
    try:
        image_url = image_generator.generate_image(prompt)
        if not image_url:
             raise ValueError("Image generation failed to return a URL.")
        safe_prompt = re.sub(r'[^\w\s-]', '', prompt).strip()
        safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt)
        save_dir = "generated_images" # Define save directory
        os.makedirs(save_dir, exist_ok=True)
        filename = f"single_{safe_prompt[:50]}.png"
        filepath = os.path.join(save_dir, filename)
        image_generator.download_image(image_url, filepath) # Download and save
        if os.path.exists(filepath):
            print(f"Image created successfully: {filepath}")
            return f"IMAGE_PATH::{filepath}" # Return path with prefix
        else:
             raise ValueError("Image downloaded but file not found at path.")
    except Exception as e:
        print(f"Error in create_single_image: {e}\n{traceback.format_exc()}")
        return f"ERROR::Failed to generate image: {str(e)}"

def generate_powerpoint_content(state: AgentState):
    """Generates content for a PowerPoint presentation."""
    topic = state.query
    retrieved_context = state.retrieved_context
    context_str = topic
    if retrieved_context:
        context_str = topic + "\n\nRelevant Context:\n" + "\n\n".join([doc.page_content for doc in retrieved_context])

    prompt_template = """Generate content for a 5-slide PowerPoint presentation about: "{topic}". Use the following context if provided and relevant:\n\nContext:\n---\n{context}\n---\nFor EACH slide (1 to 5):\n1. Provide a concise, engaging Title.\n2. Provide detailed Content (paragraphs or structured bullet points, ~3-5 points or a short paragraph).\n3. Provide a detailed Image Description suitable for DALL-E (visual, descriptive).\n\nFormat the response ONLY as:\nSlide N Title: [Title]\nSlide N Content: [Content]\nSlide N Image Description: [Description]\n--- Repeat for next slide ---"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "context"])
    response = llm.predict(prompt.format(topic=topic, context=context_str))
    return {"slide_content_response": response}

def parse_slides(state: AgentState):
    """Parses the AI-generated response into structured slides."""
    response = state.intermediate_output["slide_content_response"]
    slides = []
    pattern = re.compile(
        r"Slide\s*(?P<num>\d+)\s*Title:(?P<title>.*?)\n"
        r"Slide\s*\1\s*Content:(?P<content>.*?)\n"
        r"Slide\s*\1\s*Image Description:(?P<image_prompt>.*?)"
        r"(?=\n\s*Slide\s*\d+\s*Title:|\Z)",
        re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    matches = pattern.finditer(response)
    parsed_slides = []
    for match in matches:
        slide_data = match.groupdict()
        title = slide_data.get('title', '').strip()
        content = slide_data.get('content', '').strip()
        image_prompt = slide_data.get('image_prompt', '').strip()
        if title and content and image_prompt:
            parsed_slides.append({"title": title, "content": content, "image_prompt": image_prompt})
    return {"parsed_slides": parsed_slides}

def generate_powerpoint_images(state: AgentState):
    """Generates images for the slides."""
    parsed_slides = state.intermediate_output["parsed_slides"]
    slides_with_images = image_generator.generate_images_for_slides(parsed_slides)
    return {"slides_with_images": slides_with_images}

def create_powerpoint_file(state: AgentState):
    """Creates the PowerPoint file."""
    topic = state.query
    slides = state.intermediate_output["slides_with_images"]
    prs = Presentation()
    try:
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        slide.shapes.title.text = f"Presentation on: {topic.title()}"
        slide.placeholders[1].text = "Generated by AI Assistant"
    except Exception as e:
         print(f"Warning: Could not create title slide - {e}")

    for i, slide_data in enumerate(slides):
        try:
            slide_layout = prs.slide_layouts[5]
            slide = prs.slides.add_slide(slide_layout)
            title_shape = slide.shapes.title
            if title_shape: title_shape.text = slide_data.get("title", f"Slide {i+1}")
            content_placeholder = slide.placeholders[1]
            if content_placeholder:
                tf = content_placeholder.text_frame
                tf.text = slide_data.get("content", "")
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT
            image_placeholder = slide.placeholders[2]
            image_path = slide_data.get("image_path")
            if image_placeholder and image_path and os.path.exists(image_path):
                 try:
                    image_placeholder.insert_picture(image_path)
                 except Exception as e:
                     print(f"Error adding picture {image_path} to placeholder: {e}")
            elif image_path:
                 print(f"Image path specified ({image_path}) but file not found.")
            elif image_placeholder:
                 print(f"No image generated or path available for slide {i+1}.")
        except Exception as e:
            print(f"Error creating slide {i+1}: {e}\n{traceback.format_exc()}")

    safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()
    safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
    ppt_path = f"Presentation_{safe_topic[:50]}.pptx"
    prs.save(ppt_path)
    return {"ppt_path": f"PPT_PATH::{ppt_path}"}

def retrieve_context(state: AgentState):
    """Retrieves relevant documents from the vector database."""
    query = state.query
    results = vector_db_instance.search(query, k=5)
    return {"retrieved_context": results}

def format_response(state: AgentState):
    """Formats the final response for the user."""
    if "ppt_path" in state.intermediate_output:
        return state.intermediate_output["ppt_path"]
    elif "image_path" in state.intermediate_output:
        return state.intermediate_output["image_path"]
    elif state.intermediate_output:
        return state.intermediate_output
    else:
        return "Sorry, I couldn't process your request."

class AgenticAssistant:
    def __init__(self, vector_db, model_name="chatgpt-4o-latest", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY")):
        """Initialize the AI-powered assistant with Langgraph."""
        global vector_db_instance
        vector_db_instance = vector_db
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.image_generator = DalleImageGenerator()

        # Initialize Langgraph workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        builder = StateGraph(AgentState)

        # Add nodes for tools and actions
        builder.add_node("retrieve_context", retrieve_context)
        builder.add_node("summarize_document", summarize_document)
        builder.add_node("career_coach", career_coach)
        builder.add_node("analyze_charts", analyze_charts)
        builder.add_node("analyze_financial_data", analyze_financial_data)
        builder.add_node("answer_general_query", answer_general_query)
        builder.add_node("generate_practice_questions", generate_practice_questions)
        builder.add_node("create_single_image", create_single_image)
        builder.add_node("generate_powerpoint_content", generate_powerpoint_content)
        builder.add_node("parse_slides", parse_slides)
        builder.add_node("generate_powerpoint_images", generate_powerpoint_images)
        builder.add_node("create_powerpoint_file", create_powerpoint_file)
        builder.add_node("format_response", format_response)

        # Define edges (simplified for demonstration - needs more sophisticated routing)
        builder.set_entry_point("retrieve_context")

        # Basic linear flow for some tasks - needs more intelligent routing
        builder.add_edge("retrieve_context", "answer_general_query")
        builder.add_edge("answer_general_query", "format_response")
        builder.add_edge("retrieve_context", "summarize_document")
        builder.add_edge("summarize_document", "format_response")
        # ... add edges for other simple tools

        # Workflow for PowerPoint generation
        builder.add_edge("retrieve_context", "generate_powerpoint_content", {"condition": lambda state: "powerpoint" in state.query.lower()})
        builder.add_edge("generate_powerpoint_content", "parse_slides")
        builder.add_edge("parse_slides", "generate_powerpoint_images")
        builder.add_edge("generate_powerpoint_images", "create_powerpoint_file")
        builder.add_edge("create_powerpoint_file", "format_response")

        IMAGE_KEYWORDS = ["create", "generate", "make", "draw", "image", "picture", "drawing"]

        # Workflow for single image creation
        builder.add_edge("retrieve_context", "create_single_image", {"condition": lambda state: any(word in state.query.lower() for word in IMAGE_KEYWORDS)})
        builder.add_edge("create_single_image", "format_response")

        # Default path if no specific condition is met (e.g., general question)
        builder.add_edge("retrieve_context", "answer_general_query") # Ensure this is after more specific conditions

        builder.add_edge("format_response", END)

        return builder.compile()

    def run(self, query: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Processes the user query using the Langgraph workflow."""
        print(f"Processing query via Langgraph: {query}")
        try:
            if chat_history is None:
                chat_history = []
            inputs = {"query": query, "chat_history": chat_history}
            result = self.workflow.invoke(inputs)
            print(f"Langgraph result: {result}")
            return result['format_response']
        except Exception as e:
            print(f"Error during Langgraph execution: {e}\n{traceback.format_exc()}")
            return f"ERROR::Sorry, an error occurred while processing your request with Langgraph: {str(e)}"
