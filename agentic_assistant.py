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
from image_creator import DalleImageGenerator


class AgenticAssistant:
    def __init__(self, vector_db, model_name="chatgpt-4o-latest", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY")):
        """Initialize the AI-powered assistant."""
        self.vector_db = vector_db
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.image_generator = DalleImageGenerator()

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
                name="Image Creator",
                func=self.create_single_image,
                description="Use this tool ONLY when the user explicitly asks to create, generate, make, or draw an image, picture, or drawing based on a description. Input MUST be only the detailed description of the image itself."
            ),
            # --- POWERPOINT TOOL ---
            Tool(
                name="PowerPoint Generator",
                func=self.generate_presentation,
                description="Use this tool ONLY when the user asks to create a PowerPoint presentation, slides, or slide deck on a specific topic. Input is the topic or subject for the presentation."
            )
        ]
        return tools

    def run(self, query: str) -> str:
        """
        Processes the user query using the conversational agent,
        which selects and runs the appropriate tool.
        """
        print(f"Processing query via Agent: {query}")
        try:
            # Directly use the Langchain agent executor
            # The agent will use tool descriptions to route the query
            # Make sure tool descriptions are clear and distinct
            response = self.agent.run(query) # Use agent.run with initialize_agent
            return response
        except Exception as e:
            print(f"Error during agent execution: {e}\n{traceback.format_exc()}")
            # Return error with prefix for Streamlit handling
            return f"ERROR::Sorry, an error occurred while processing your request: {str(e)}"

    
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

    # --- NEW Method for Single Image Creation ---
    def create_single_image(self, prompt: str) -> str:
        """Generates a single image using DalleImageGenerator based on the prompt."""
        if not self.image_generator:
            return "ERROR::Image generator is not available."

        print(f"Tool 'Image Creator' called with prompt: {prompt[:100]}...")
        try:
            # Use the DalleImageGenerator instance
            # generate_image should return URL, download_image saves it and returns path
            image_url = self.image_generator.generate_image(prompt)
            if not image_url:
                 raise ValueError("Image generation failed to return a URL.")

            # Create a safe filename based on the prompt
            safe_prompt = re.sub(r'[^\w\s-]', '', prompt).strip()
            safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt)
            save_dir = "generated_images" # Define save directory
            filename = f"single_{safe_prompt[:50]}.png"
            filepath = os.path.join(save_dir, filename)

            self.image_generator.download_image(image_url, filepath) # Download and save

            if os.path.exists(filepath):
                print(f"Image created successfully: {filepath}")
                return f"IMAGE_PATH::{filepath}" # Return path with prefix
            else:
                 raise ValueError("Image downloaded but file not found at path.")

        except Exception as e:
            print(f"Error in create_single_image: {e}\n{traceback.format_exc()}")
            return f"ERROR::Failed to generate image: {str(e)}"

    # --- Modified PowerPoint Generation ---
    def generate_presentation(self, topic: str, num_slides: int = 5, use_vector_db: bool = True) -> str:
        """Generates a PowerPoint presentation with AI-generated content and images."""
        if not self.image_generator:
            return "ERROR::Image generator is not available for presentation."

        print(f"Tool 'PowerPoint Generator' called for topic: {topic}")
        context = topic
        if use_vector_db:
            print("Retrieving context...")
            try:
                results = self.vector_db.search(topic, k=3) # Reduced k for context brevity
                if results:
                    context = topic + "\n\nRelevant Context:\n" + "\n\n".join([doc.page_content for doc in results if hasattr(doc, 'page_content')])
                    print(f"Using context from {len(results)} retrieved documents.")
                else:
                    print("No relevant documents found, using topic as context.")
            except Exception as e:
                print(f"Error searching vector database: {e}. Using topic as context.")

        # Step 2: Generate structured slide content using LLM
        print(f"Generating content for {num_slides} slides...")
        prompt_for_slides = f"""
Generate content for a {num_slides}-slide PowerPoint presentation about: "{topic}".
Use the following context if provided and relevant:
Context:
---
{context}
---
For EACH slide (1 to {num_slides}):
1.  Provide a concise, engaging Title.
2.  Provide detailed Content (paragraphs or structured bullet points, ~3-5 points or a short paragraph).
3.  Provide a detailed Image Description suitable for DALL-E (visual, descriptive).

Format the response ONLY as:
Slide N Title: [Title]
Slide N Content: [Content]
Slide N Image Description: [Description]
--- Repeat for next slide ---"""
        try:
            # Use the agent's LLM (ChatOpenAI) directly via predict for this structured task
            slide_content_response = self.llm.predict(prompt_for_slides)
            # Parse the text response into structured slide data
            parsed_slides = self._parse_slides(slide_content_response)
            print(f"Successfully parsed {len(parsed_slides)} slides from LLM response.")
            if not parsed_slides:
                 return "ERROR::Failed to parse slide content from the language model."
            # Limit to requested number of slides if LLM gives more
            parsed_slides = parsed_slides[:num_slides]

        except Exception as e:
            print(f"Error generating slide content with LLM: {e}\n{traceback.format_exc()}")
            return f"ERROR::Error generating slide content: {str(e)}"

        # Step 3: Generate images for the parsed slides using DalleImageGenerator
        print("Generating images for slides...")
        try:
            # This method in DalleImageGenerator handles generating, downloading, and adding paths
            slides_with_images = self.image_generator.generate_images_for_slides(parsed_slides)
            print(f"Image generation attempted for {len(slides_with_images)} slides.")
            # Check for errors reported by the image generator
            errors = [s.get('error') for s in slides_with_images if 'error' in s]
            if errors:
                 print(f"Warning: Encountered errors during image generation: {errors}")

        except Exception as e:
            print(f"Error during batch image generation: {e}\n{traceback.format_exc()}")
            # Proceed without images or return error? Let's proceed without.
            slides_with_images = parsed_slides # Use original parsed slides if batch generation failed

        # Step 4: Create PowerPoint file
        print("Creating PowerPoint file...")
        try:
            ppt_path = self._create_pptx(topic, slides_with_images) # Pass slides possibly containing image_path
            print(f"PowerPoint file saved to: {ppt_path}")
            return f"PPT_PATH::{ppt_path}" # Return path with prefix
        except Exception as e:
            print(f"Error creating PowerPoint file: {e}\n{traceback.format_exc()}")
            return f"ERROR::Error creating PowerPoint file: {str(e)}"

    def _parse_slides(self, response: str) -> list:
        """Parses the AI-generated response into structured slides using regex."""
        # (Using the more robust regex implementation from previous correction)
        print("Parsing slide content...")
        slides = []
        pattern = re.compile(
            r"Slide\s*(?P<num>\d+)\s*Title:(?P<title>.*?)\n"
            r"Slide\s*\1\s*Content:(?P<content>.*?)\n"
            r"Slide\s*\1\s*Image Description:(?P<image_prompt>.*?)"
            r"(?=\n\s*Slide\s*\d+\s*Title:|\Z)",
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )
        matches = pattern.finditer(response)
        count = 0
        for match in matches:
            count += 1
            slide_data = match.groupdict()
            title = slide_data.get('title', '').strip()
            content = slide_data.get('content', '').strip()
            image_prompt = slide_data.get('image_prompt', '').strip()
            if title and content and image_prompt:
                slides.append({
                    "title": title,
                    "content": content,
                    "image_prompt": image_prompt # Keep prompt for potential use
                })
            else:
                print(f"Warning: Incomplete data parsed for slide block near match {count}.")
        print(f"Parsed {len(slides)} slides using regex.")
        return slides


    # --- REMOVED _generate_slide_image method ---
    # Image generation is now handled by DalleImageGenerator instance

    def _create_pptx(self, topic: str, slides: list) -> str:
        """Creates and formats a PowerPoint file using slide data (incl. image paths)."""
        prs = Presentation()
        # Add Title Slide
        try:
            title_slide_layout = prs.slide_layouts[0]
            slide = prs.slides.add_slide(title_slide_layout)
            slide.shapes.title.text = f"Presentation on: {topic.title()}"
            slide.placeholders[1].text = "Generated by AI Assistant"
        except Exception as e:
             print(f"Warning: Could not create title slide - {e}")

        # Add Content Slides
        for i, slide_data in enumerate(slides):
            print(f"  Adding Slide {i+1} to PPT: {slide_data.get('title', 'No Title')[:50]}...")
            try:
                # Use layout 5 (Title and Two Content) for text + image side-by-side
                slide_layout = prs.slide_layouts[5]
                slide = prs.slides.add_slide(slide_layout)

                # Add title
                title_shape = slide.shapes.title
                if title_shape: title_shape.text = slide_data.get("title", f"Slide {i+1}")

                # Add text content
                content_placeholder = slide.placeholders[1] # Left content placeholder
                if content_placeholder:
                    tf = content_placeholder.text_frame
                    tf.text = slide_data.get("content", "")
                    tf.word_wrap = True # Enable word wrap
                    tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT # Adjust shape size to text
                    # Optional: Adjust font size if needed
                    # for paragraph in tf.paragraphs:
                    #     for run in paragraph.runs:
                    #         run.font.size = Pt(12)

                # Add image if path exists
                image_placeholder = slide.placeholders[2] # Right content placeholder
                image_path = slide_data.get("image_path") # Path added by generate_images_for_slides
                if image_placeholder and image_path and os.path.exists(image_path):
                     print(f"    Adding image {image_path}...")
                     try:
                        # Add picture, maintaining aspect ratio within placeholder bounds
                        image_placeholder.insert_picture(image_path)
                     except Exception as e:
                         print(f"    Error adding picture {image_path} to placeholder: {e}")
                elif image_path:
                     print(f"    Image path specified ({image_path}) but file not found.")
                elif image_placeholder:
                     print(f"    No image generated or path available for slide {i+1}.")

            except Exception as e:
                print(f"  Error creating slide {i+1}: {e}\n{traceback.format_exc()}")

        # Sanitize topic for filename
        safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
        ppt_path = f"Presentation_{safe_topic[:50]}.pptx"

        prs.save(ppt_path)
        return ppt_path
