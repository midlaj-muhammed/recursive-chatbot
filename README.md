# 🤖 Enhanced BERT Question Answering Chatbot

A high-performance, hybrid AI chatbot capable of answering complex questions from uploaded documents (PDF, TXT, DOCX).

It combines the **speed and privacy** of local BERT models with the **reasoning power** of Google Gemini API.

## 🚀 Key Features

### 1. Hybrid AI Architecture
-   **⚡ Local Speed**: Uses `distilbert-base-cased-distilled-squad` for lightning-fast, privacy-focused answers.
-   **🧠 Cloud Intelligence**: Optional integration with **Google Gemini API** (`gemini-flash-latest`) to refinement and polish answers.
-   **🔄 Recursive Language Model (RLM)**: A specialized mode for complex questions. It breaks them down into sub-queries, solves each piece, and synthesizes a comprehensive answer.

### 2. Document Processing
-   **Files**: Supports PDF, TXT, and DOCX files.
-   **Smart Chunking**: Automatically splits long documents into overlapping segments (2500 chars) to handle large files without losing context.
-   **Context Enrichment**: Automatically expands short answers to full sentences for better readability.

### 3. User Interface
-   **Interactive Dashboard**: Built with Streamlit for a clean, responsive experience.
-   **Real-time Settings**: Toggle "Enhance with Gemini" and "Recursive Mode" instantly.
-   **Confidence Scores**: Visual indicators (High/Medium/Low) for answer reliability.

---

## 🛠️ Installation

### Prerequisites
-   Python 3.8+
-   A Google Gemini API Key (Optional, for advanced features)

### Setup Steps

1.  **Clone the repository**:
    ```bash
    cd "/home/midlaj/Documents/IBM chatbot"
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key (Recommended)**:
    -   Create a `.env` file in the project directory.
    -   Add your Gemini API key:
        ```env
        GEMINI_API_KEY=your_api_key_here
        ```
    -   *Note: Without this key, the app works in "Local BERT Only" mode.*

---

## 💻 Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Upload a Document**:
    -   Use the sidebar to upload a PDF, DOCX, or TXT file.

3.  **Ask a Question**:
    -   Type your question in the main text box.

4.  **Advanced Modes**:
    -   **✨ Enhance with Gemini**: Turn this toggle ON to let Gemini rephrase BERT's answer into natural language.
    -   **🧠 Recursive Mode (RLM)**: (Requires 'Enhance' to be ON). Turn this ON for complex multi-part questions.
        -   *Example*: "What is the main idea of the document and what evidence supports it?"
        -   The bot will: Decompose -> Solve Sub-questions -> Synthesize Final Answer.

---

## 🏗️ Project Architecture

```mermaid
graph TD
    A[User Question] --> B{Mode Selected?}
    B -->|Standard| C[Local DistilBERT]
    B -->|Enhance| D[Local DistilBERT]
    B -->|Recursive (RLM)| E[Gemini Decomposition]
    
    C --> F[Raw Extraction]
    D --> F
    F --> G[Gemini Refinement]
    
    E --> H[Sub-Question 1]
    E --> I[Sub-Question 2]
    H --> C
    I --> C
    C --> J[Synthesize Evidence]
    J --> G
    
    G --> K[Final Answer]
    F --> K
```

## 🔧 Technical Details

-   **Model**: `distilbert-base-cased-distilled-squad` (40% smaller, 60% faster than standard BERT).
-   **Gemini Model**: `gemini-flash-latest` used for refinement and reasoning.
-   **Caching**: Streamlit's `@st.cache_resource` is used to load the models only once, ensuring fast reloads.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `AttributeError: 'QAEngine' object has no attribute 'gemini_enabled'` | Click **"🔄 Reload Engine"** in the sidebar to refresh the cache. |
| `Gemini key not found` | Ensure your `.env` file is saved and contains `GEMINI_API_KEY`. Click "Reload Engine". |
| `429 Quota Exceeded` | You may have hit the free tier limit for Gemini. Wait a minute and try again. |
| **Answers are cut off** | Increase the "Max Answer Length" slider in the sidebar. |

## 📦 Dependencies

-   `streamlit`: Web UI
-   `transformers` & `torch`: AI Models
-   `google-generativeai`: Gemini API
-   `python-dotenv`: Security
-   `PyPDF2` & `python-docx`: File parsing

---

