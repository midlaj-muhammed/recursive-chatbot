"""
Question Answering Engine using BERT
Handles question answering using pre-trained BERT model from Hugging Face.
"""

from transformers import pipeline
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class QAEngine:
    """Enhanced BERT-based Question Answering Engine with Gemini refinement."""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize the QA engine with a pre-trained DistilBERT model and Gemini.
        """
        print(f"Loading Optimization Model: {model_name}...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name
        )
        self.max_length = 512
        
        # Initialize Gemini
        self.gemini_enabled = False
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-flash-latest')
                self.gemini_enabled = True
                print("Gemini API initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize Gemini: {str(e)}")
        else:
            print("Gemini API key not found. Gemini refinement will be disabled.")
            
        print("Model loaded successfully!")
    
    def answer_question(
        self, 
        question: str, 
        context: str,
        max_answer_length: int = 100,
        enhance_with_gemini: bool = False,
        use_rlm: bool = False
    ) -> Dict[str, any]:
        """
        Answer a question with optimized speed, optional Gemini refinement, or full RLM.
        """
        if not question or not question.strip():
            return {'answer': 'Please provide a question.', 'score': 0.0}
        
        if not context or not context.strip():
            return {'answer': 'No document context provided.', 'score': 0.0}

        # RLM Mode (Recursive Language Model)
        if use_rlm and self.gemini_enabled:
            return self._answer_with_rlm(question, context, max_answer_length)
        
        # BERT-based retrieval
        if len(context) > 3000:
            chunks = self._chunk_text(context, chunk_size=2500, overlap=300)
            result = self._answer_from_chunks(question, chunks, max_answer_length)
        else:
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=max_answer_length
                )
            except Exception as e:
                return {'answer': f'Error: {str(e)}', 'score': 0.0}

        # Context enrichment (Local)
        if result['score'] > 0.05:
            result['answer'] = self._enrich_answer_with_context(result['answer'], context)
            
            # Gemini Refinement (Hybrid)
            if enhance_with_gemini and self.gemini_enabled:
                result['answer'] = self._refine_with_gemini(question, result['answer'], context)
            
        return result

    def _answer_with_rlm(self, question: str, context: str, max_answer_length: int) -> Dict[str, any]:
        """
        Executes the Recursive Language Model strategy: Decompose -> Solve -> Synthesize.
        """
        try:
            # 1. Decompose
            sub_questions = self._decompose_question(question)
            print(f"RLM Sub-questions: {sub_questions}")
            
            # 2. Recursive Solve
            evidence = []
            for sub_q in sub_questions:
                # Use local BERT to solve each sub-question
                # Recurse: answer_question -> BERT retrieval
                res = self.answer_question(sub_q, context, max_answer_length, enhance_with_gemini=False, use_rlm=False)
                if res['score'] > 0.01: # Only keep relevant evidence
                    evidence.append(f"Q: {sub_q}\nFound: {res['answer']}")
            
            if not evidence:
                return {'answer': "I couldn't find sufficient information to answer complex inquiry.", 'score': 0.0}
                
            # 3. Synthesize
            final_answer = self._synthesize_answers(question, evidence)
            return {'answer': final_answer, 'score': 1.0} # Score 1.0 because it's synthesized
            
        except Exception as e:
            print(f"RLM Error: {str(e)}")
            # Fallback to standard method
            return self.answer_question(question, context, max_answer_length, enhance_with_gemini=True, use_rlm=False)

    def _decompose_question(self, question: str) -> List[str]:
        """Uses Gemini to break a complex question into simple sub-questions."""
        prompt = f"""
        Break down this complex question into 2-4 simple, independent sub-questions that can be answered using a document.
        Return ONLY the questions, one per line.
        
        Complex Question: {question}
        """
        response = self.gemini_model.generate_content(prompt)
        text = response.text.strip()
        questions = [q.strip("- ").strip() for q in text.split('\n') if q.strip()]
        return questions

    def _synthesize_answers(self, original_question: str, evidence: List[str]) -> str:
        """Uses Gemini to combine evidence into a final answer."""
        evidence_text = "\n\n".join(evidence)
        prompt = f"""
        System: Synthesize a comprehensive answer to the user's question based ONLY on the evidence provided below.
        
        User Question: {original_question}
        
        Evidence Collected:
        {evidence_text}
        
        Final Answer:
        """
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()

    def _refine_with_gemini(self, question: str, bert_answer: str, context: str) -> str:
        """
        Uses Gemini to refine and improve theBERT-extracted answer using context.
        """
        try:
            # We provide a short snippet of the context to keep it fast/within limits if possible
            # But here we use the bert_answer as a base
            prompt = f"""
            System: You are an AI assistant helping to refine answers extracted from a document.
            Context snippet from document: {bert_answer}
            
            Question: {question}
            
            Based on the context snippet, provide a more comprehensive, natural, and well-phrased answer. 
            If the context doesn't contain enough info, just improve the phrasing of the existing answer.
            Be concise but thorough.
            """
            
            response = self.gemini_model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            return bert_answer
        except Exception as e:
            print(f"Gemini refinement failed: {str(e)}")
            return bert_answer

    def _enrich_answer_with_context(self, short_answer: str, full_context: str) -> str:
        """
        Expands a short answer by including the full sentence it belongs to, 
        making the answer feel more natural and complete.
        """
        if not short_answer or len(short_answer.split()) > 15:
            return short_answer

        # Find the start index of the answer in the context
        idx = full_context.find(short_answer)
        if idx == -1: return short_answer

        # Look for the sentence boundary before the answer
        start = max(0, full_context.rfind('.', 0, idx) + 1)
        # Look for the sentence boundary after the answer
        end = full_context.find('.', idx + len(short_answer))
        if end == -1: end = len(full_context)
        else: end += 1 # Include the period
        
        enriched = full_context[start:end].strip()
        
        # If the enriched version is exactly the same or too long, just return the original/cleaned
        if len(enriched) < len(short_answer): return short_answer
        
        return enriched

    def _chunk_text(self, text: str, chunk_size: int = 2500, overlap: int = 300) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
    
    def _answer_from_chunks(
        self, 
        question: str, 
        chunks: List[str],
        max_answer_length: int = 100
    ) -> Dict[str, any]:
        """Find the best answer across chunks."""
        best_result = {'answer': 'Could not find an answer.', 'score': 0.0}
        
        for chunk in chunks:
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=chunk,
                    max_answer_len=max_answer_length
                )
                if result['score'] > best_result['score']:
                    best_result = result
            except:
                continue
        return best_result
    
    def get_multiple_answers(
        self,
        question: str,
        context: str,
        top_k: int = 3
    ) -> List[Dict[str, any]]:
        """
        Get multiple possible answers to a question.
        
        Args:
            question: The question to answer
            context: The context/document to search
            top_k: Number of answers to return
            
        Returns:
            List of top k answers
        """
        if len(context) > 3000:
            chunks = self._chunk_text(context, chunk_size=2500, overlap=200)
            all_answers = []
            
            for chunk in chunks:
                try:
                    result = self.qa_pipeline(
                        question=question,
                        context=chunk,
                        top_k=top_k
                    )
                    if isinstance(result, list):
                        all_answers.extend(result)
                    else:
                        all_answers.append(result)
                except Exception:
                    continue
            
            # Sort by score and return top k
            all_answers.sort(key=lambda x: x['score'], reverse=True)
            return all_answers[:top_k]
        
        try:
            result = self.qa_pipeline(
                question=question,
                context=context,
                top_k=top_k
            )
            return result if isinstance(result, list) else [result]
        except Exception as e:
            return [{
                'answer': f'Error: {str(e)}',
                'score': 0.0,
                'start': 0,
                'end': 0
            }]
