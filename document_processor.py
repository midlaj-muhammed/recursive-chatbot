"""
Document Processor Module
Handles document upload and text extraction from various file formats.
Supports: PDF, TXT, DOCX
"""

import os
from typing import Optional
import PyPDF2
from docx import Document


class DocumentProcessor:
    """Process and extract text from uploaded documents."""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def process_document(self, file_path: str) -> Optional[str]:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text as string, or None if processing fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {self.supported_formats}")
        
        try:
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_ext == '.txt':
                return self._extract_from_txt(file_path)
            elif file_ext == '.docx':
                return self._extract_from_docx(file_path)
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return self._clean_text(text)
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self._clean_text(text)
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def process_uploaded_file(self, uploaded_file) -> Optional[str]:
        """
        Process file uploaded through Streamlit.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Extracted text as string
        """
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            text = self.process_document(temp_path)
            
            return text
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
