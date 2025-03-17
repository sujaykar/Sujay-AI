#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from typing import List, Dict, Any
import tempfile
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def process_file(self, file_path: str) -> List[Document]:
        """Process a file based on its extension and return document chunks."""
        _, file_extension = os.path.splitext(file_path)
        
        try:
            if file_extension.lower() == '.pdf':
                return self._process_pdf(file_path)
            elif file_extension.lower() == '.txt':
                return self._process_text(file_path)
            elif file_extension.lower() in ['.png', '.jpg', '.jpeg']:
                return self._process_image(file_path)
            elif file_extension.lower() == '.md':
                return self._process_markdown(file_path)
            elif file_extension.lower() in ['.xlsx', '.xls']:
                return self._process_excel(file_path)
            elif file_extension.lower() == '.csv':
                return self._process_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def _process_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _process_text(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
   def _process_image(self, file_path: str) -> List[Document]:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            documents = [Document(page_content=text, metadata={"source": file_path})]
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error processing image with OCR: {str(e)}")
            # Return minimal document with error info
            documents = [Document(
                page_content=f"[Image processing error: {str(e)}]", 
                metadata={"source": file_path}
            )]
            return documents
            
    def _process_markdown(self, file_path: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _process_excel(self, file_path: str) -> List[Document]:
        loader = UnstructuredExcelLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _process_csv(self, file_path: str) -> List[Document]:
        loader = CSVLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Process a file uploaded through Streamlit."""
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name
        
        # Process the temporary file
        documents = self.process_file(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return documents


# In[ ]:




