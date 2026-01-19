"""
BERT Question Answering Chatbot
A Streamlit web application for document-based question answering using BERT.
"""

import streamlit as st
from document_processor import DocumentProcessor
from qa_engine import QAEngine
import os

# Page configuration
st.set_page_config(
    page_title="BERT Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        color: #0e1117;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'document_text' not in st.session_state:
    st.session_state.document_text = None
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'qa_engine' not in st.session_state:
    st.session_state.qa_engine = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Initialize processors
@st.cache_resource
def load_qa_engine():
    """Load the QA engine (cached to avoid reloading)."""
    return QAEngine()

def get_confidence_class(score):
    """Return CSS class based on confidence score."""
    if score >= 0.7:
        return "confidence-high"
    elif score >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_confidence_label(score):
    """Return human-readable confidence label."""
    if score >= 0.7:
        return "High Confidence"
    elif score >= 0.4:
        return "Medium Confidence"
    else:
        return "Low Confidence"

# Main app
def main():
    # Header
    st.title("🤖 BERT Question Answering Chatbot")
    st.markdown("Upload a document and ask questions about its content using AI-powered BERT model.")
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload a PDF, TXT, or DOCX file"
        )
        
        if uploaded_file is not None:
            # Process document if it's new
            if st.session_state.document_name != uploaded_file.name:
                with st.spinner("Processing document..."):
                    try:
                        doc_processor = DocumentProcessor()
                        text = doc_processor.process_uploaded_file(uploaded_file)
                        
                        st.session_state.document_text = text
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.qa_history = []  # Clear history for new document
                        
                        st.success(f"✅ Document '{uploaded_file.name}' loaded successfully!")
                        st.info(f"📊 Document length: {len(text)} characters")
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        st.session_state.document_text = None
                        st.session_state.document_name = None
            else:
                st.success(f"✅ Document '{uploaded_file.name}' is loaded")
                st.info(f"📊 Document length: {len(st.session_state.document_text)} characters")
        
        # Document preview
        if st.session_state.document_text:
            st.divider()
            st.subheader("📖 Document Preview")
            with st.expander("View document content"):
                st.text_area(
                    "Document Text",
                    st.session_state.document_text,
                    height=300,
                    disabled=True,
                    label_visibility="collapsed"
                )
        
        # Settings
        st.divider()
        st.subheader("⚙️ Settings")
        
        # Gemini Enhancement toggle
        enhance_with_gemini = st.toggle(
            "✨ Enhance with Gemini",
            value=False,
            help="Use Gemini API to rephrase and improve answers (requires API Key)"
        )
        
        # Check Gemini status
        qa_engine = load_qa_engine()
        gemini_ready = getattr(qa_engine, 'gemini_enabled', False)
        
        if enhance_with_gemini:
            if gemini_ready:
                st.success("💎 Gemini Refinement Active")
            else:
                st.warning("⚠️ Gemini key not found in .env")

        # RLM Toggle (Only if Gemini is active)
        use_rlm = False
        if enhance_with_gemini and gemini_ready:
            use_rlm = st.toggle(
                "🧠 Recursive Mode (RLM)",
                value=False,
                help="Deconstructs complex questions into sub-parts (Slower but detailed)"
            )
            if use_rlm:
                st.info("RLM: Decompose -> Solve -> Synthesize")
                
        max_answer_length = st.slider(
            "Max Answer Length",
            min_value=10,
            max_value=200,
            value=100,
            help="Maximum number of words in the answer"
        )
        
        show_confidence = st.checkbox("Show Confidence Score", value=True)
        
        if st.button("🔄 Reload Engine"):
            st.cache_resource.clear()
            st.session_state.qa_engine = None
            st.success("Engine cache cleared. Reloading...")
            st.rerun()
            
        if st.button("🗑️ Clear History"):
            st.session_state.qa_history = []
            st.rerun()
    
    # Main content area
    if st.session_state.document_text is None:
        st.info("👈 Please upload a document from the sidebar to get started.")
        
        # Show example
        st.markdown("---")
        st.subheader("📚 How to use:")
        st.markdown("""
        1. **Upload a document** (PDF, TXT, or DOCX) using the sidebar
        2. **Ask questions** about the document content
        3. **Get AI-powered answers** from the BERT model
        
        **Example questions you can ask:**
        - What is the main topic of this document?
        - Who are the key people mentioned?
        - When did the events occur?
        - Where did this take place?
        - Why is this important?
        """)
    else:
        # Question input
        st.subheader("❓ Ask a Question")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the main topic of this document?",
                label_visibility="collapsed"
            )
        
        with col2:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        # Process question
        if ask_button or (question and question.strip()):
            if question and question.strip():
                with st.spinner("🤔 Thinking..."):
                    # Load QA engine
                    if st.session_state.qa_engine is None:
                        st.session_state.qa_engine = load_qa_engine()
                    
                    # Get answer
                    try:
                        result = st.session_state.qa_engine.answer_question(
                            question=question,
                            context=st.session_state.document_text,
                            max_answer_length=max_answer_length,
                            enhance_with_gemini=enhance_with_gemini,
                            use_rlm=use_rlm
                        )
                    except TypeError:
                        # Handle cached object mismatch (old engine version)
                        st.cache_resource.clear()
                        st.session_state.qa_engine = load_qa_engine()
                        result = st.session_state.qa_engine.answer_question(
                            question=question,
                            context=st.session_state.document_text,
                            max_answer_length=max_answer_length,
                            enhance_with_gemini=enhance_with_gemini,
                            use_rlm=use_rlm
                        )
                    
                    # Add to history
                    st.session_state.qa_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'score': result['score']
                    })
        
        # Display Q&A history
        if st.session_state.qa_history:
            st.markdown("---")
            st.subheader("💬 Q&A History")
            
            # Display in reverse order (newest first)
            for i, qa in enumerate(reversed(st.session_state.qa_history)):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.qa_history) - i}:** {qa['question']}")
                    
                    # Answer box
                    confidence_class = get_confidence_class(qa['score'])
                    confidence_label = get_confidence_label(qa['score'])
                    
                    st.markdown(f"""
                        <div class="answer-box">
                            <strong>Answer:</strong> {qa['answer']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if show_confidence:
                        st.markdown(
                            f"<span class='{confidence_class}'>Confidence: {qa['score']:.2%} ({confidence_label})</span>",
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()
