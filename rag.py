import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union
from pathlib import Path
import traceback
import time
import hashlib
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    UnstructuredWordDocumentLoader, JSONLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus, FAISS
from langchain.schema.retriever import BaseRetriever
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA
try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

# Additional imports
import pymilvus
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, 
    TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
)
import torch
import warnings
from threading import Thread
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
try:
    from guidance import models, gen, select
    GUIDANCE_AVAILABLE = True
except ImportError:
    GUIDANCE_AVAILABLE = False
    print("⚠️ Guidance not available. Install with: pip install guidance")

warnings.filterwarnings("ignore")

@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion techniques"""
    use_synonyms: bool = True
    use_hypernyms: bool = True
    use_related_terms: bool = True
    max_expansions: int = 3

@dataclass
class ChunkingConfig:
    """Configuration for advanced chunking strategies"""
    strategy: str = "semantic"  # "semantic", "recursive", "adaptive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic_threshold: float = 0.5
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

class JSONStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for JSON generation"""
    
    def __init__(self, tokenizer, max_brackets: int = 100):
        self.tokenizer = tokenizer
        self.max_brackets = max_brackets
        self.bracket_count = 0
        self.in_json = False
        self.json_complete = False
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the last few tokens to check for JSON completion
        if input_ids.shape[1] < 2:
            return False
            
        last_tokens = input_ids[0, -10:].tolist()
        decoded = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        
        # Count brackets to detect JSON completion
        open_brackets = decoded.count('{') + decoded.count('[')
        close_brackets = decoded.count('}') + decoded.count(']')
        
        # Check if we have a complete JSON structure
        if open_brackets > 0:
            self.in_json = True
        
        if self.in_json and open_brackets == close_brackets and open_brackets > 0:
            # Check if the JSON ends properly (no additional text)
            try:
                # Try to find the last complete JSON structure
                json_match = re.search(r'(\{.*\}|\[.*\])', decoded, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    json.loads(json_str)  # Validate JSON
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass
        
        return False

class AdvancedPromptTemplate:
    """Advanced prompt engineering with multiple strategies"""
    
    @staticmethod
    def get_json_prompt(context: str, question: str, schema: Dict[str, Any] = None) -> str:
        """Generate a controlled prompt for JSON responses"""
        schema_instruction = ""
        if schema:
            schema_instruction = f"\nThe JSON must follow this exact schema: {json.dumps(schema, indent=2)}"
        
        return f"""You are a precise AI assistant that returns ONLY valid JSON responses. No explanations, no markdown, no additional text.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Return ONLY valid JSON
- No text before or after the JSON
- No explanations or comments
- No markdown formatting{schema_instruction}

JSON RESPONSE:"""

    @staticmethod
    def get_analytical_prompt(context: str, question: str) -> str:
        """Generate a prompt for analytical responses"""
        return f"""You are an expert analyst. Provide a comprehensive, structured analysis based on the given context.

CONTEXT:
{context}

QUESTION: {question}

ANALYSIS FRAMEWORK:
1. Direct Answer: Provide the specific answer to the question
2. Supporting Evidence: Reference specific parts of the context
3. Implications: Discuss broader implications or consequences
4. Confidence Level: Rate your confidence (High/Medium/Low) with reasoning

STRUCTURED RESPONSE:"""

    @staticmethod
    def get_conversational_prompt(context: str, question: str) -> str:
        """Generate a prompt for conversational responses"""
        return f"""You are a knowledgeable assistant having a natural conversation. Use the provided context to give helpful, accurate answers.

CONTEXT:
{context}

QUESTION: {question}

Please provide a clear, helpful response that directly answers the question while being conversational and engaging:"""

    @staticmethod
    def get_detailed_analysis_prompt(context: str, question: str) -> str:
        """Generate a specialized prompt for detailed analysis with strict JSON format"""
        return f"""You are a domain expert. Analyze the provided information and return ONLY valid JSON.

CONTEXT:
{context}

TASK: {question}

CRITICAL INSTRUCTIONS:
- Return ONLY the JSON object below
- Do NOT add any explanations, comments, or markdown
- Do NOT return arrays - return the EXACT object structure shown
- Include both "entities" and "relationships" properties

REQUIRED JSON FORMAT (return EXACTLY this structure):
{{
  "entities": [
    {{
      "id": "entity_id",
      "name": "entity name",
      "description": "detailed description of the entity",
      "attributes": ["list", "of", "attributes"]
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "relationship type",
      "description": "description of relationship"
    }}
  ],
  "summary": "overall summary of findings"
}}

JSON RESPONSE:"""

class SemanticChunker:
    """Advanced semantic chunking based on sentence embeddings"""
    
    def __init__(self, embeddings_model, config: ChunkingConfig):
        self.embeddings_model = embeddings_model
        self.config = config
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using semantic similarity"""
        if self.config.strategy == "semantic":
            return self._semantic_chunking(documents)
        elif self.config.strategy == "adaptive":
            return self._adaptive_chunking(documents)
        else:
            # Fallback to recursive chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            return splitter.split_documents(documents)
    
    def _semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """Implement semantic chunking based on sentence similarity"""
        chunked_docs = []
        
        for doc in documents:
            # Split into sentences
            sentences = re.split(r'[.!?]+', doc.page_content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                continue
            
            # Get embeddings for sentences
            try:
                sentence_embeddings = self.embeddings_model.embed_documents(sentences)
            except Exception as e:
                print(f"⚠️ Error in semantic chunking: {e}, falling back to recursive")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                return splitter.split_documents(documents)
            
            # Group sentences by semantic similarity
            chunks = self._group_sentences_by_similarity(
                sentences, sentence_embeddings, self.config.semantic_threshold
            )
            
            # Create document chunks
            for i, chunk_sentences in enumerate(chunks):
                chunk_content = '. '.join(chunk_sentences)
                if len(chunk_content) >= self.config.min_chunk_size:
                    chunk_doc = Document(
                        page_content=chunk_content,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i,
                            'chunk_method': 'semantic',
                            'chunk_size': len(chunk_content)
                        }
                    )
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _group_sentences_by_similarity(self, sentences: List[str], embeddings: List[List[float]], threshold: float) -> List[List[str]]:
        """Group sentences by semantic similarity"""
        if not sentences:
            return []
        
        groups = [[sentences[0]]]
        current_group_embedding = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]
            
            # Calculate similarity with current group
            group_avg_embedding = np.mean(current_group_embedding, axis=0)
            similarity = cosine_similarity(
                [sentence_embedding], [group_avg_embedding]
            )[0][0]
            
            # Check chunk size constraints
            current_group_size = len('. '.join(groups[-1]))
            
            if (similarity > threshold and 
                current_group_size + len(sentence) < self.config.max_chunk_size):
                # Add to current group
                groups[-1].append(sentence)
                current_group_embedding.append(sentence_embedding)
            else:
                # Start new group
                groups.append([sentence])
                current_group_embedding = [sentence_embedding]
        
        return groups
    
    def _adaptive_chunking(self, documents: List[Document]) -> List[Document]:
        """Adaptive chunking based on content type and structure"""
        chunked_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Detect content type and adapt chunking strategy
            if self._is_code_content(content):
                chunks = self._chunk_code_content(content)
            elif self._is_structured_content(content):
                chunks = self._chunk_structured_content(content)
            else:
                chunks = self._chunk_narrative_content(content)
            
            # Create document objects
            for i, chunk in enumerate(chunks):
                if len(chunk) >= self.config.min_chunk_size:
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i,
                            'chunk_method': 'adaptive',
                            'chunk_size': len(chunk)
                        }
                    )
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _is_code_content(self, content: str) -> bool:
        """Detect if content is primarily code"""
        code_indicators = ['{', '}', 'function', 'class', 'def', 'import', 'pragma']
        return sum(indicator in content for indicator in code_indicators) > 3
    
    def _is_structured_content(self, content: str) -> bool:
        """Detect if content has clear structure (headers, lists, etc.)"""
        structure_indicators = ['#', '##', '1.', '2.', '-', '*', ':']
        return sum(line.strip().startswith(indicator) for line in content.split('\n') 
                  for indicator in structure_indicators) > len(content.split('\n')) * 0.1
    
    def _chunk_code_content(self, content: str) -> List[str]:
        """Chunk code content by functions/classes"""
        # Simple function-based chunking for code
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            current_chunk.append(line)
            if (line.strip().startswith(('function', 'def', 'class')) and 
                len('\n'.join(current_chunk)) > self.config.chunk_size):
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_structured_content(self, content: str) -> List[str]:
        """Chunk structured content by sections"""
        # Split by headers and structural elements
        sections = re.split(r'\n(?=#+\s|\d+\.\s|[-*]\s)', content)
        return [section.strip() for section in sections if section.strip()]
    
    def _chunk_narrative_content(self, content: str) -> List[str]:
        """Chunk narrative content by paragraphs and sentences"""
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class QueryExpander:
    """Advanced query expansion for better retrieval"""
    
    def __init__(self, config: QueryExpansionConfig):
        self.config = config
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded_queries = [query]  # Always include original
        
        if self.config.use_synonyms:
            expanded_queries.extend(self._get_synonyms(query))
        
        if self.config.use_related_terms:
            expanded_queries.extend(self._get_related_terms(query))
        
        # Limit expansions
        return expanded_queries[:self.config.max_expansions + 1]
    
    def _get_synonyms(self, query: str) -> List[str]:
        """Get synonym-based expansions"""
        # Simple synonym mapping for general terms
        synonym_map = {
            'issue': ['problem', 'concern', 'defect', 'bug'],
            'function': ['method', 'procedure', 'routine', 'operation'],
            'document': ['file', 'record', 'text', 'content'],
            'important': ['critical', 'essential', 'significant', 'key'],
            'error': ['mistake', 'fault', 'bug', 'failure'],
            'analysis': ['examination', 'evaluation', 'assessment', 'review'],
            'feature': ['capability', 'functionality', 'aspect', 'property']
        }
        
        synonyms = []
        words = query.lower().split()
        
        for word in words:
            if word in synonym_map:
                for synonym in synonym_map[word]:
                    new_query = query.lower().replace(word, synonym)
                    synonyms.append(new_query)
        
        return synonyms[:2]  # Limit synonym expansions
    
    def _get_related_terms(self, query: str) -> List[str]:
        """Get related term expansions"""
        # Add context-specific terms based on query content
        if 'performance' in query.lower() or 'optimization' in query.lower():
            return [
                query + " efficiency analysis",
                query + " performance metrics"
            ]
        elif 'error' in query.lower():
            return [
                query + " error handling",
                query + " bug detection"
            ]
        elif 'function' in query.lower():
            return [
                query + " function analysis",
                query + " method implementation"
            ]
        elif 'data' in query.lower():
            return [
                query + " data processing",
                query + " information analysis"
            ]
        
        return []

class ControlledGenerator:
    """Controlled generation for structured outputs"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.guidance_model = None
        
        if GUIDANCE_AVAILABLE:
            try:
                # Check if model is a string (model name) or actual model object
                if isinstance(model, str):
                    self.guidance_model = models.Transformers(model, tokenizer=tokenizer)
                else:
                    # For actual model objects, we'll skip guidance for now
                    # as it requires more complex setup
                    print("ℹ️ Guidance integration requires model name string, using fallback generation")
                    self.guidance_model = None
            except Exception as e:
                print(f"⚠️ Could not initialize guidance model: {e}")
                self.guidance_model = None
    
    def generate_json(self, prompt: str, schema: Dict[str, Any] = None, max_tokens: int = 1024) -> Dict[str, Any]:
        """Generate JSON with controlled output"""
        if self.guidance_model and GUIDANCE_AVAILABLE:
            return self._guided_json_generation(prompt, schema, max_tokens)
        else:
            return self._constrained_json_generation(prompt, max_tokens)
    
    def _guided_json_generation(self, prompt: str, schema: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Use guidance for controlled JSON generation"""
        try:
            with self.guidance_model.chat():
                self.guidance_model += prompt
                self.guidance_model += gen('json_response', regex=r'\{.*\}', max_tokens=max_tokens)
            
            json_str = self.guidance_model['json_response']
            return json.loads(json_str)
        except Exception as e:
            print(f"⚠️ Guided generation failed: {e}, falling back to constrained generation")
            return self._constrained_json_generation(prompt, max_tokens)
    
    def _constrained_json_generation(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Generate JSON with custom stopping criteria"""
        try:
            # Get the device of the model
            model_device = next(self.model.parameters()).device
            
            # Tokenize input and move to model device
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Create stopping criteria for JSON
            stopping_criteria = StoppingCriteriaList([
                JSONStoppingCriteria(self.tokenizer)
            ])
            
            # Generate with constraints
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and extract JSON
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()
            
            # Extract JSON from response
            return self._extract_clean_json(response_text)
            
        except Exception as e:
            print(f"❌ Constrained generation failed: {e}")
            return {"error": f"Generation failed: {e}"}
    
    def _extract_clean_json(self, text: str) -> Dict[str, Any]:
        """Extract clean JSON from potentially messy text"""
        # Remove common prefixes/suffixes
        text = re.sub(r'^[^{[]*', '', text)  # Remove text before JSON
        text = re.sub(r'[^}\]]*$', '', text)  # Remove text after JSON
        
        # Find JSON boundaries - try multiple patterns
        json_patterns = [
            r'(\{[^{}]*"enriched_nodes"[^{}]*"semantic_edges"[^{}]*\})',  # Look for enriched_nodes schema
            r'(\{.*?"enriched_nodes".*?"semantic_edges".*?\})',  # More flexible enriched_nodes pattern
            r'(\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\})',  # Nested objects
            r'(\[[^\[\]]*\[[^\[\]]*\][^\[\]]*\]|\[[^\[\]]*\])',  # Nested arrays
            r'(\{.*?\})',  # Simple objects
            r'(\[.*?\])'   # Simple arrays
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Try to parse the JSON
                    parsed = json.loads(match)
                    
                    # If we get a simple array but expect enriched_nodes schema, 
                    # try to transform it into the expected format
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Check if this looks like it should be enriched_nodes format
                        if all(isinstance(item, str) for item in parsed):
                            # Convert array of vulnerabilities to proper schema
                            transformed = {
                                "enriched_nodes": [],
                                "semantic_edges": []
                            }
                            # Add a note that this was transformed
                            print("ℹ️ Transformed simple array to enriched_nodes schema")
                            return transformed
                    
                    return parsed
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, try to fix common issues
        try:
            # Remove trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
            # Add missing quotes around keys
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            return {"error": "Could not extract valid JSON", "raw_text": text}

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses"""
    
    def __init__(self, callback_func=None):
        self.callback_func = callback_func
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM produces a new token"""
        self.tokens.append(token)
        if self.callback_func:
            self.callback_func(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes"""
        if self.callback_func:
            self.callback_func(None, is_end=True)

class DocumentReranker:
    """Cross-encoder based document reranker for improved relevance"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        try:
            print(f"🔄 Loading reranker model: {model_name}")
            self.cross_encoder = CrossEncoder(model_name)
            self.model_name = model_name
            print(f"✅ Reranker model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load reranker model: {e}")
            print("⚠️ Continuing without reranker...")
            self.cross_encoder = None
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on query-document relevance scores
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not self.cross_encoder or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores
            scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Combine documents with scores and sort
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores[:top_k]
            
        except Exception as e:
            print(f"❌ Error in reranking: {e}")
            # Fallback to original order
            return [(doc, 0.0) for doc in documents[:top_k]]

class EnhancedRerankingRetriever(BaseRetriever):
    """Enhanced retriever that combines query expansion and reranking"""
    
    vector_store: Any
    reranker: Any  
    query_expander: Optional[Any]
    retrieval_k: int
    final_k: int
    
    def __init__(self, vector_store, reranker, query_expander, retrieval_k: int, final_k: int):
        super().__init__(
            vector_store=vector_store,
            reranker=reranker,
            query_expander=query_expander,
            retrieval_k=retrieval_k,
            final_k=final_k
        )
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Required method for BaseRetriever"""
        return self.get_relevant_documents(query)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents using query expansion and reranking"""
        if self.query_expander:
            # Use query expansion
            expanded_queries = self.query_expander.expand_query(query)
            all_docs = []
            
            for q in expanded_queries:
                docs = self.vector_store.similarity_search(q, k=self.retrieval_k // len(expanded_queries))
                all_docs.extend(docs)
            
            # Remove duplicates based on content
            unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
            print(f"🔍 Query expansion: {len(expanded_queries)} queries → {len(all_docs)} docs → {len(unique_docs)} unique")
        else:
            # Standard retrieval
            unique_docs = self.vector_store.similarity_search(query, k=self.retrieval_k)
        
        if not unique_docs:
            return []
        
        # Apply reranking
        if self.reranker:
            reranked_docs_with_scores = self.reranker.rerank_documents(query, unique_docs, self.final_k)
            reranked_docs = []
            for doc, score in reranked_docs_with_scores:
                doc.metadata['rerank_score'] = float(score)
                reranked_docs.append(doc)
            
            print(f"🔄 Enhanced reranking: {len(unique_docs)} → {len(reranked_docs)} documents")
            return reranked_docs
        else:
            return unique_docs[:self.final_k]

class RAGSystem:
    def __init__(self, 
                 milvus_uri: str = "",
                 milvus_user: str = "",
                 milvus_password: str = "",
                 collection_name: str = "document_embeddings",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 llm_model: str = "Qwen/Qwen2.5-3B-Instruct",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 use_milvus: bool = False,
                 use_reranker: bool = True,
                 use_controlled_generation: bool = True,
                 use_semantic_chunking: bool = True,
                 use_query_expansion: bool = True,
                 use_multi_gpu: bool = False,
                 quantization_bits: Optional[int] = None,
                 retrieval_k: int = 10,  # Retrieve more documents for reranking
                 final_k: int = 3,       # Final number of documents after reranking
                 chunking_config: Optional[ChunkingConfig] = None,
                 query_expansion_config: Optional[QueryExpansionConfig] = None):
        
        self.milvus_uri = milvus_uri
        self.milvus_user = milvus_user
        self.milvus_password = milvus_password
        self.collection_name = collection_name
        self.use_milvus = use_milvus
        self.use_reranker = use_reranker
        self.use_controlled_generation = use_controlled_generation
        self.use_semantic_chunking = use_semantic_chunking
        self.use_query_expansion = use_query_expansion
        self.use_multi_gpu = use_multi_gpu
        self.quantization_bits = quantization_bits
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        self.vector_store = None
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.qa_chain = None
        self.reranker = None
        self.controlled_generator = None
        self.semantic_chunker = None
        self.query_expander = None
        
        # Multi-GPU setup
        self.device_map = None
        self.num_gpus = 0
        self.gpu_devices = []
        
        # Set up configurations
        self.chunking_config = chunking_config or ChunkingConfig()
        self.query_expansion_config = query_expansion_config or QueryExpansionConfig()
        
        # Initialize GPU configuration
        self._setup_gpu_configuration()
        
        print("🚀 Initializing Advanced RAG System with Enhanced Features...")
        print(f"   🔄 Reranking: {'✅' if use_reranker else '❌'}")
        print(f"   🎯 Controlled Generation: {'✅' if use_controlled_generation else '❌'}")
        print(f"   🧠 Semantic Chunking: {'✅' if use_semantic_chunking else '❌'}")
        print(f"   🔍 Query Expansion: {'✅' if use_query_expansion else '❌'}")
        print(f"   🖥️ Multi-GPU: {'✅' if use_multi_gpu and self.num_gpus > 1 else '❌'}")
        if self.use_multi_gpu:
            print(f"   📊 Available GPUs: {self.num_gpus}")
            print(f"   🎮 GPU Devices: {self.gpu_devices}")
        
        # Initialize embeddings model
        print("📚 Loading embedding model...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            print(f"✅ Embedding model loaded: {embedding_model}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
        
        # Initialize reranker
        if use_reranker:
            self.reranker = DocumentReranker(reranker_model)
        
        # Initialize query expander
        if use_query_expansion:
            self.query_expander = QueryExpander(self.query_expansion_config)
        
        # Initialize semantic chunker
        if use_semantic_chunking:
            self.semantic_chunker = SemanticChunker(self.embeddings, self.chunking_config)
        else:
            # Initialize standard text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunking_config.chunk_size,
                chunk_overlap=self.chunking_config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Initialize LLM with enhanced capabilities
        print("🤖 Loading LLM model with advanced capabilities...")
        self._initialize_advanced_llm(llm_model)
        
        # Automatically initialize QA chain if vector store and LLM are ready
        print("🔗 Initializing QA chain...")
        self._initialize_qa_chain()
        
        print("✅ Advanced RAG System initialized successfully!")
    
    def _setup_gpu_configuration(self):
        """Setup multi-GPU configuration"""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_devices = [f"cuda:{i}" for i in range(self.num_gpus)]
            
            if self.use_multi_gpu and self.num_gpus > 1:
                print(f"🖥️ Multi-GPU mode enabled with {self.num_gpus} GPUs")
                
                # Create device map for model parallelism
                self.device_map = self._create_device_map()
                
                # Set CUDA_VISIBLE_DEVICES if not set
                if "CUDA_VISIBLE_DEVICES" not in os.environ:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(self.num_gpus)])
                    
            elif self.use_multi_gpu and self.num_gpus == 1:
                print("⚠️ Multi-GPU requested but only 1 GPU available, using single GPU")
                self.use_multi_gpu = False
                self.device_map = {"": "cuda:0"}
            else:
                self.device_map = "auto" if self.num_gpus > 0 else None
        else:
            print("⚠️ No CUDA GPUs available, using CPU")
            self.use_multi_gpu = False
            self.num_gpus = 0
            self.device_map = None

    def _create_device_map(self) -> Dict[str, Union[int, str]]:
        """Create device map for multi-GPU model parallelism"""
        if not self.use_multi_gpu or self.num_gpus <= 1:
            return "auto"
        
        # Strategy: Distribute model layers across available GPUs
        device_map = {}
        
        if self.num_gpus == 2:
            # For 2 GPUs: split roughly in half
            device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.layers.4": 0,
                "model.layers.5": 0,
                "model.layers.6": 0,
                "model.layers.7": 0,
                "model.layers.8": 1,
                "model.layers.9": 1,
                "model.layers.10": 1,
                "model.layers.11": 1,
                "model.layers.12": 1,
                "model.layers.13": 1,
                "model.layers.14": 1,
                "model.layers.15": 1,
                "model.norm": 1,
                "lm_head": 1
            }
        elif self.num_gpus >= 4:
            # For 4+ GPUs: more granular distribution
            layers_per_gpu = 32 // self.num_gpus  # Assuming ~32 layers
            device_map = {"model.embed_tokens": 0}
            
            for layer_idx in range(32):
                gpu_idx = min(layer_idx // layers_per_gpu, self.num_gpus - 1)
                device_map[f"model.layers.{layer_idx}"] = gpu_idx
            
            device_map.update({
                "model.norm": self.num_gpus - 1,
                "lm_head": self.num_gpus - 1
            })
        else:
            # For 3 GPUs or other configurations, use auto
            device_map = "auto"
        
        print(f"🎮 Created device map for {self.num_gpus} GPUs")
        return device_map

    def _get_embedding_device(self) -> str:
        """Get the appropriate device for embeddings model"""
        if self.use_multi_gpu and self.num_gpus > 1:
            # Use the first GPU for embeddings to avoid conflicts
            return "cuda:0"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _get_memory_usage_per_gpu(self) -> Dict[int, Dict[str, float]]:
        """Get memory usage statistics for each GPU"""
        if not torch.cuda.is_available():
            return {}
        
        gpu_stats = {}
        for i in range(self.num_gpus):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            gpu_stats[i] = {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "total_gb": total_memory,
                "utilization_percent": (memory_allocated / total_memory) * 100
            }
        
        return gpu_stats

    def _initialize_vector_store(self):
        """Initialize vector store with proper error handling"""
        if not self.use_milvus:
            print("📁 Using FAISS vector store (local)")
            self.vector_store = None
            return
    
        try:
            print("🔗 Connecting to Milvus Cloud...")
            connections.disconnect("default")
            connections.connect(
                alias="default",
                uri=self.milvus_uri,
                user=self.milvus_user,
                password=self.milvus_password,
                secure=True
            )
            print("✅ Connected to Milvus Cloud successfully!")
    
            if utility.has_collection(self.collection_name):
                print(f"📋 Collection '{self.collection_name}' already exists")
                collection = Collection(self.collection_name)
                collection.load()
                schema = collection.schema
                print(f"🔍 Collection schema: {schema}")
                field_names = [field.name for field in schema.fields]
                print(f"🔍 Existing collection schema fields: {field_names}")
    
                required_fields = {'vector', 'text', 'metadata'}
                existing_fields = set(field_names)
                if not required_fields.issubset(existing_fields):
                    print(f"⚠️ Incompatible schema. Required: {required_fields}, Found: {existing_fields}")
                    print("🔄 Falling back to FAISS due to schema incompatibility...")
                    self.use_milvus = False
                    self.vector_store = None
                    return
    
                self.vector_store = Milvus(
                    embedding_function=self.embeddings,
                    connection_args={
                        "uri": self.milvus_uri,
                        "user": self.milvus_user,
                        "password": self.milvus_password,
                        "secure": True
                    },
                    collection_name=self.collection_name
                )
                print(f"✅ Loaded existing collection: {self.collection_name}")
                
                # Check if collection has documents and initialize QA chain if it does
                if collection.num_entities > 0:
                    print(f"📚 Collection contains {collection.num_entities} documents")
                    self._initialize_qa_chain()
    
            else:
                print(f"🆕 Creating new collection: {self.collection_name}")
                self._create_milvus_collection()
    
        except Exception as e:
            print(f"❌ Error connecting to Milvus: {e}")
            print(f"📝 Full error: {traceback.format_exc()}")
            print("🔄 Falling back to local FAISS vector store...")
            self.use_milvus = False
            self.vector_store = None
    
    def _create_milvus_collection(self):
        """Create a new Milvus collection with proper schema"""
        try:
            # Define collection schema - dimension should match embedding model
            embedding_dim = 1024  # BAAI/bge-large-en-v1.5 dimension
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            schema = CollectionSchema(
                fields=fields, 
                description="Smart contract document embeddings for RAG"
            )
            
            # Create collection
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            # Create index for efficient similarity search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            
            print(f"✅ Created collection '{self.collection_name}' with index")
            
            # Initialize vector store
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                connection_args={
                    "uri": self.milvus_uri,
                    "user": self.milvus_user,
                    "password": self.milvus_password,
                    "secure": True
                },
                collection_name=self.collection_name
            )
            
        except Exception as e:
            print(f"❌ Error creating Milvus collection: {e}")
            raise
    
    def _initialize_advanced_llm(self, model_name: str):
        """Initialize the language model with advanced features and multi-GPU support"""
        try:
            gpu_info = ""
            if self.use_multi_gpu and self.num_gpus > 1:
                gpu_info = f" with {self.num_gpus} GPUs"
            elif torch.cuda.is_available():
                gpu_info = f" with 1 GPU"
            else:
                gpu_info = " with CPU"
                
            print(f"🤖 Loading {model_name}{gpu_info} with {'no quantization' if self.quantization_bits is None else f'{self.quantization_bits}-bit quantization'}...")
            
            # Configure quantization if specified
            quantization_config = None
            if self.quantization_bits in [4, 8]:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=(self.quantization_bits == 4),
                        load_in_8bit=(self.quantization_bits == 8),
                        bnb_4bit_quant_type="nf4" if self.quantization_bits == 4 else None,
                        bnb_4bit_compute_dtype=torch.float16 if self.quantization_bits == 4 else None,
                        bnb_4bit_use_double_quant=True if self.quantization_bits == 4 else False
                    )
                    print(f"✅ Quantization configured: {self.quantization_bits}-bit")
                except ImportError as e:
                    print(f"❌ bitsandbytes not installed: {e}")
                    print("🔄 Proceeding without quantization...")
                    quantization_config = None
            
            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Configure model loading parameters
            model_kwargs = {
                "quantization_config": quantization_config,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Set up multi-GPU or single GPU configuration
            if self.use_multi_gpu and self.num_gpus > 1:
                print(f"🖥️ Configuring model for {self.num_gpus} GPUs...")
                model_kwargs.update({
                    "device_map": self.device_map,
                    "torch_dtype": torch.float16,
                    "max_memory": self._calculate_max_memory_per_gpu()
                })
                
                # Enable model parallelism
                if hasattr(torch.nn, 'DataParallel'):
                    print("⚡ Multi-GPU model parallelism enabled")
                    
            elif torch.cuda.is_available():
                print("🎮 Configuring model for single GPU...")
                model_kwargs.update({
                    "device_map": "auto",
                    "torch_dtype": torch.float16 if quantization_config is None else torch.float32
                })
            else:
                print("💻 Configuring model for CPU...")
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                    "device_map": None
                })
            
            # Load model
            print("🔄 Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Display GPU memory usage
            if self.use_multi_gpu and self.num_gpus > 1:
                self._display_gpu_memory_usage()
            
            # Initialize controlled generator
            if self.use_controlled_generation:
                print("🎯 Initializing controlled generation...")
                self.controlled_generator = ControlledGenerator(self.model, self.tokenizer)
                print("✅ Controlled generation initialized")
            
            # Create text generation pipeline with improved settings
            print("⚙️ Creating generation pipeline...")
            pipe_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_new_tokens": 4096,
                "temperature": 0.1,
                "batch_size": 1,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Adjust batch size for multi-GPU
            if self.use_multi_gpu and self.num_gpus > 1:
                pipe_kwargs["batch_size"] = min(self.num_gpus, 4)  # Optimize batch size
            
            pipe = pipeline("text-generation", **pipe_kwargs)
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            print(f"✅ Successfully loaded: {model_name}")
            if self.use_multi_gpu and self.num_gpus > 1:
                print(f"🚀 Model distributed across {self.num_gpus} GPUs")
            
        except Exception as model_error:
            print(f"⚠️ Failed to load {model_name}: {model_error}")
            print("🔄 Trying fallback model...")
            
            # Fallback to a smaller, more reliable model
            fallback_model = "microsoft/DialoGPT-medium"
            try:
                print(f"📱 Loading fallback model: {fallback_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                
                # Simpler configuration for fallback
                fallback_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "low_cpu_mem_usage": True
                }
                
                if torch.cuda.is_available():
                    fallback_kwargs["device_map"] = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(fallback_model, **fallback_kwargs)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Initialize controlled generator for fallback model too
                if self.use_controlled_generation:
                    self.controlled_generator = ControlledGenerator(self.model, self.tokenizer)
                
                pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=1024,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                print(f"✅ Successfully loaded fallback model: {fallback_model}")
                
            except Exception as fallback_error:
                print(f"❌ Fallback model also failed: {fallback_error}")
                raise

    def _calculate_max_memory_per_gpu(self) -> Dict[int, str]:
        """Calculate maximum memory allocation per GPU"""
        if not torch.cuda.is_available() or not self.use_multi_gpu:
            return {}
        
        max_memory = {}
        for i in range(self.num_gpus):
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(i).total_memory
            # Reserve 85% for model, 15% for operations
            usable_memory = int(total_memory * 0.85)
            max_memory[i] = f"{usable_memory // (1024**3)}GB"
        
        print(f"💾 GPU memory allocation: {max_memory}")
        return max_memory

    def _display_gpu_memory_usage(self):
        """Display current GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        print("📊 GPU Memory Usage:")
        gpu_stats = self._get_memory_usage_per_gpu()
        for gpu_id, stats in gpu_stats.items():
            print(f"   GPU {gpu_id}: {stats['allocated_gb']:.1f}GB/{stats['total_gb']:.1f}GB "
                  f"({stats['utilization_percent']:.1f}% utilized)")

    def _initialize_qa_chain(self):
        """Initialize the QA chain with advanced prompt engineering"""
        if not self.llm:
            print("⚠️ LLM not loaded yet - QA chain cannot be initialized")
            return
            
        if not self.vector_store:
            print("⚠️ No vector store available - QA chain will use direct generation mode")
            print("ℹ️ Add documents with rag.ingest_file('path/to/document') to enable context-based queries")
            # Still set qa_chain to None so queries fall back to direct generation
            self.qa_chain = None
            return
            
        try:
            # Check if vector store has documents
            has_documents = False
            document_count = 0
            
            if self.use_milvus:
                try:
                    collection = Collection(self.collection_name)
                    document_count = collection.num_entities
                    has_documents = document_count > 0
                except Exception as e:
                    print(f"⚠️ Could not check Milvus collection: {e}")
                    has_documents = True  # Assume documents exist to proceed with QA chain
            else:
                # For FAISS, assume documents exist if vector store is initialized
                has_documents = True
                document_count = "unknown (FAISS)"
            
            if not has_documents:
                print(f"⚠️ Vector store has no documents ({document_count} entities) - QA chain will use direct generation")
                print("ℹ️ Add documents with rag.ingest_file('path/to/document') to enable context-based queries")
                self.qa_chain = None
                return
            
            # Create a generic prompt template
            prompt_template = """You are a helpful AI assistant. Analyze the provided context and answer the question with precision and clarity.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide accurate, context-based responses
- If asked for JSON format, return ONLY valid JSON without additional text
- Be specific and reference the context when possible
- Use clear, professional language

RESPONSE:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create enhanced retriever with query expansion and reranking
            retriever = self._create_enhanced_retriever()
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print(f"✅ QA chain initialized successfully! ({document_count} documents available)")
            if self.use_reranker and self.reranker:
                print(f"   🔄 Advanced reranking enabled with {self.reranker.model_name}")
            if self.use_query_expansion and self.query_expander:
                print(f"   🔍 Query expansion enabled")
            
        except Exception as e:
            print(f"❌ Error initializing QA chain: {e}")
            print(f"📝 Full error: {traceback.format_exc()}")
            print("⚠️ Falling back to direct generation mode")
            self.qa_chain = None

    def _create_enhanced_retriever(self) -> BaseRetriever:
        """Create an enhanced retriever with multiple optimization techniques"""
        if self.use_reranker and self.reranker:
            return EnhancedRerankingRetriever(
                vector_store=self.vector_store,
                reranker=self.reranker,
                query_expander=self.query_expander if self.use_query_expansion else None,
                retrieval_k=self.retrieval_k,
                final_k=self.final_k
            )
        else:
            return self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.final_k}
            )

    def _compute_document_hash(self, document: Document) -> str:
        """Compute a unique hash for a document based on content and metadata"""
        content = document.page_content
        metadata = json.dumps(document.metadata, sort_keys=True)
        combined = content + metadata
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _check_existing_documents(self, documents: List[Document]) -> List[Document]:
        """Check for existing documents in the vector store and return new documents"""
        if not self.use_milvus or not utility.has_collection(self.collection_name):
            return documents
        
        try:
            collection = Collection(self.collection_name)
            collection.load()
            
            # Get existing document metadata
            results = collection.query(expr="id >= 0", output_fields=["metadata"], limit=10000)
            existing_hashes = set()
            existing_sources = set()
            
            for result in results:
                metadata = json.loads(result['metadata'])
                source = metadata.get('source', '')
                existing_sources.add(source)
                # Compute hash of existing document if needed
                if 'content_hash' in metadata:
                    existing_hashes.add(metadata['content_hash'])
            
            new_documents = []
            for doc in documents:
                doc_hash = self._compute_document_hash(doc)
                source = doc.metadata.get('source', '')
                
                # Skip if document source or hash already exists
                if source in existing_sources or doc_hash in existing_hashes:
                    print(f"ℹ️ Skipping document from {source}: already exists in collection")
                    continue
                
                doc.metadata['content_hash'] = doc_hash
                new_documents.append(doc)
            
            return new_documents
            
        except Exception as e:
            print(f"❌ Error checking existing documents: {e}")
            return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents with advanced chunking strategies"""
        if not documents:
            return []
        
        try:
            if self.use_semantic_chunking and self.semantic_chunker:
                print("🧠 Using semantic chunking...")
                texts = self.semantic_chunker.chunk_documents(documents)
            else:
                print("📝 Using recursive chunking...")
                texts = self.text_splitter.split_documents(documents)
            
            # Add enhanced metadata
            for i, text in enumerate(texts):
                text.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(text.page_content),
                    'processing_timestamp': time.time(),
                    'chunking_strategy': self.chunking_config.strategy if self.use_semantic_chunking else 'recursive'
                })
            
            print(f"✅ Processed into {len(texts)} chunks using {texts[0].metadata.get('chunking_strategy', 'unknown')} strategy")
            return texts
            
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            return []

    def query(self, 
              question: str,
              response_type: str = "conversational",
              schema: Optional[Dict[str, Any]] = None,
              use_controlled_generation: Optional[bool] = None,
              generation_length: Optional[int] = None,
              max_retries: int = 3,
              use_streaming: bool = False,
              callback_func=None,
              use_direct_generation: bool = False,
              expect_json: bool = None) -> Union[Dict[str, Any], Iterator[str]]:
        """
        Unified query method that handles all types of queries
        
        Args:
            question: Your question about smart contracts or any topic
            response_type: "json", "analytical", "conversational" (default: "conversational")
            schema: JSON schema for structured responses (optional)
            use_controlled_generation: Override default controlled generation setting (optional)
            generation_length: Maximum number of tokens to generate (optional, auto-set based on response_type)
            max_retries: Number of retries for JSON generation (default: 3)
            use_streaming: Stream response token by token (default: False)
            callback_func: Callback function for streaming (optional)
            use_direct_generation: Force use of direct generation bypassing QA chain (default: False)
            expect_json: Backward compatibility - sets response_type to "json" if True (deprecated)
            
        Returns:
            Dict with response data, or Iterator[str] if streaming
        """
        
        # Handle backward compatibility
        if expect_json is True:
            response_type = "json"
            print("⚠️ 'expect_json' parameter is deprecated, use response_type='json' instead")
        
        # Set default generation length based on response type
        if generation_length is None:
            if response_type == "json":
                generation_length = 1024  # Shorter for JSON responses
            elif response_type == "analytical":
                generation_length = 3072  # Longer for analytical responses
            else:
                generation_length = 2048  # Standard for conversational
        
        # Validate generation length
        generation_length = max(64, min(generation_length, 8192))
        
        # Handle streaming requests
        if use_streaming:
            return self._handle_streaming_query(question, generation_length, callback_func)
        
        # Handle direct generation requests
        if use_direct_generation:
            return self._handle_direct_generation_query(question, response_type, generation_length, schema)
        
        # Handle standard enhanced queries
        return self._handle_enhanced_query(
            question=question,
            response_type=response_type,
            schema=schema,
            use_controlled_generation=use_controlled_generation,
            generation_length=generation_length,
            max_retries=max_retries
        )

    def _handle_streaming_query(self, question: str, generation_length: int, callback_func=None) -> Iterator[str]:
        """Handle streaming query requests"""
        if not self.qa_chain:
            yield "❌ System not ready. Please ensure vector store is initialized and contains documents."
            return
        
        try:
            print(f"🔍 Streaming query: {question[:100]}...")
            print(f"📏 Stream generation length: {generation_length} tokens")
            
            # Get relevant documents first (this is fast with reranking)
            if self.use_reranker and self.reranker:
                retriever = EnhancedRerankingRetriever(
                    vector_store=self.vector_store,
                    reranker=self.reranker,
                    query_expander=self.query_expander if self.use_query_expansion else None,
                    retrieval_k=self.retrieval_k,
                    final_k=self.final_k
                )
                docs = retriever.get_relevant_documents(question)
            else:
                docs = self.vector_store.similarity_search(question, k=self.final_k)
            
            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Prepare the prompt
            prompt_template = """You are a smart contract security expert. Based on the provided context, answer the question with a precise, structured response.

CONTEXT:
{context}

QUESTION: {question}

IMPORTANT INSTRUCTIONS:
- If the question asks for JSON format, return ONLY valid JSON without any additional text, explanations, or markdown
- If the question asks for specific analysis, provide exactly what is requested
- Be concise and direct in your response
- Focus on smart contract security aspects when relevant

ANSWER:"""
            
            prompt = prompt_template.format(context=context, question=question)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            
            # Get model device and move inputs to same device
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Create streamer
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                timeout=30.0, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            # Generate response in a separate thread
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": generation_length,
                "temperature": 0.1,
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield tokens as they come
            for token in streamer:
                if callback_func:
                    callback_func(token)
                yield token
            
            thread.join()
            
        except Exception as e:
            error_msg = f"❌ Streaming query failed: {e}"
            print(error_msg)
            yield error_msg

    def _handle_direct_generation_query(self, question: str, response_type: str, generation_length: int, schema: Optional[Dict]) -> Dict[str, Any]:
        """Handle direct generation query requests"""
        try:
            print(f"🔧 Using direct generation with {generation_length} tokens")
            
            # Get relevant documents
            docs = self._get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create appropriate prompt based on response type
            if response_type == "json":
                if schema and 'enriched_nodes' in schema.get('properties', {}):
                    prompt = AdvancedPromptTemplate.get_detailed_analysis_prompt(context, question)
                else:
                    prompt = AdvancedPromptTemplate.get_json_prompt(context, question, schema)
            elif response_type == "analytical":
                prompt = AdvancedPromptTemplate.get_analytical_prompt(context, question)
            else:
                prompt = AdvancedPromptTemplate.get_conversational_prompt(context, question)
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_length,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):].strip()
            
            # Handle JSON extraction if needed
            parsed_json = None
            is_valid_json = False
            if response_type == "json":
                parsed_json = self._extract_json_from_response(answer)
                is_valid_json = parsed_json is not None
            
            return {
                "question": question,
                "answer": answer,
                "parsed_json": parsed_json,
                "is_valid_json": is_valid_json,
                "response_type": response_type,
                "generation_method": "direct",
                "generation_length": generation_length,
                "source_documents": self._format_source_docs(docs),
                "num_sources_used": len(docs),
                "success": True
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": "",
                "parsed_json": None,
                "is_valid_json": False,
                "response_type": response_type,
                "generation_method": "direct",
                "generation_length": generation_length,
                "error": str(e),
                "success": False
            }

    def _handle_enhanced_query(self, question: str, response_type: str, schema: Optional[Dict], 
                              use_controlled_generation: Optional[bool], generation_length: int, max_retries: int) -> Dict[str, Any]:
        """Handle enhanced query requests with fallback logic"""
        
        use_controlled = use_controlled_generation if use_controlled_generation is not None else self.use_controlled_generation
        
        if not self.qa_chain:
            return {
                "error": "System not ready. Please ensure vector store is initialized and contains documents.",
                "question": question,
                "answer": "",
                "response_type": response_type,
                "generation_length": generation_length
            }
        
        try:
            print(f"🔍 Processing query: {question[:100]}...")
            print(f"📏 Generation length: {generation_length} tokens")
            start_time = time.time()
            
            # Handle JSON with controlled generation
            if response_type == "json" and use_controlled and self.controlled_generator:
                result = self._handle_controlled_json_query(question, schema, max_retries, generation_length)
                result["query_time"] = time.time() - start_time
                return result
            
            # Try standard query with custom generation length
            try:
                result = self._handle_standard_query(question, response_type, generation_length)
                result["query_time"] = time.time() - start_time
                
                # If JSON was requested but we didn't get valid JSON, try direct generation
                if response_type == "json" and not result.get("is_valid_json", False):
                    print("🔄 Standard query didn't produce valid JSON, trying direct generation...")
                    direct_result = self._handle_direct_generation_query(question, response_type, generation_length, schema)
                    if direct_result.get("is_valid_json", False):
                        direct_result["query_time"] = time.time() - start_time
                        direct_result["fallback_used"] = True
                        return direct_result
                
                return result
                
            except Exception as e:
                print(f"⚠️ Standard query failed: {e}")
                if "RetrievalQA" in str(e) or "chain" in str(e).lower():
                    print("🔄 Chain issue detected, using direct generation...")
                    direct_result = self._handle_direct_generation_query(question, response_type, generation_length, schema)
                    direct_result["query_time"] = time.time() - start_time
                    direct_result["fallback_used"] = True
                    return direct_result
                else:
                    raise
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {
                "error": f"Query failed: {e}",
                "question": question,
                "answer": "",
                "response_type": response_type,
                "generation_length": generation_length,
                "query_time": time.time() - start_time if 'start_time' in locals() else 0
            }


    def _handle_controlled_json_query(self, question: str, schema: Optional[Dict], max_retries: int, generation_length: int) -> Dict[str, Any]:
        """Handle JSON queries with controlled generation"""
        print("🎯 Using controlled JSON generation...")
        
        # Get relevant documents
        docs = self._get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Check if this is a detailed analysis query
        is_detailed_analysis = any(keyword in question.lower() for keyword in [
            'analyze', 'detailed', 'entities', 'relationships',
            'comprehensive', 'network', 'connections', 'structure'
        ])
        
        # Create appropriate prompt
        if is_detailed_analysis and schema and 'entities' in schema.get('properties', {}):
            print("🔧 Using specialized detailed analysis prompt")
            prompt = AdvancedPromptTemplate.get_detailed_analysis_prompt(context, question)
        else:
            print("🔧 Using general JSON prompt")
            prompt = AdvancedPromptTemplate.get_json_prompt(context, question, schema)
        
        # Try controlled generation with retries
        for attempt in range(max_retries):
            try:
                print(f"🔄 JSON generation attempt {attempt + 1}/{max_retries}")
                
                result = self.controlled_generator.generate_json(prompt, schema, generation_length)
                
                if "error" not in result:
                    print("✅ Successfully generated controlled JSON")
                    
                    # Validate against schema if provided
                    if schema:
                        validation = self.validate_json_schema(result, schema)
                        if not validation.get('valid'):
                            print(f"⚠️ Schema validation failed: {validation.get('errors', [])}")
                            
                            # If it's entities schema and we got wrong format, try to fix
                            if 'entities' in schema.get('properties', {}):
                                if isinstance(result, list):
                                    print("🔧 Converting array to entities format")
                                    result = {
                                        "entities": [],
                                        "relationships": [],
                                        "summary": "Automatically converted from list format"
                                    }
                    
                    return {
                        "question": question,
                        "answer": json.dumps(result, indent=2),
                        "parsed_json": result,
                        "is_valid_json": True,
                        "response_type": "json",
                        "generation_method": "controlled",
                        "attempts_used": attempt + 1,
                        "source_documents": self._format_source_docs(docs),
                        "query_time": time.time()
                    }
                else:
                    print(f"⚠️ Controlled generation failed on attempt {attempt + 1}: {result.get('error')}")
                    
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
        
        # Fallback to standard query if controlled generation fails
        print("🔄 Falling back to standard query with JSON extraction...")
        fallback_question = f"{question}\n\nIMPORTANT: Return ONLY valid JSON in your response. No explanations, no markdown, no additional text."
        return self._handle_standard_query(fallback_question, "json", generation_length)

    def _handle_standard_query(self, question: str, response_type: str, generation_length: int) -> Dict[str, Any]:
        """Handle standard queries with appropriate prompting"""
        start_time = time.time()
        
        # Apply query optimization
        optimized_question = self._optimize_query_for_model(question)
        
        # Try to create a custom pipeline with generation length
        custom_pipeline_used = False
        original_llm = None
        
        try:
            # Enhanced chain structure detection
            llm_access_paths = [
                # Try different possible LLM locations in the chain
                ('combine_documents_chain.llm_chain.llm', lambda: self.qa_chain.combine_documents_chain.llm_chain.llm),
                ('combine_documents_chain.llm', lambda: self.qa_chain.combine_documents_chain.llm),
                ('llm_chain.llm', lambda: self.qa_chain.llm_chain.llm),
                ('llm', lambda: self.qa_chain.llm),
            ]
            
            temp_llm = None
            llm_setter = None
            
            # Try each path to find and access the LLM
            for path_name, llm_getter in llm_access_paths:
                try:
                    original_llm = llm_getter()
                    if original_llm is not None:
                        print(f"🔧 Found LLM at: {path_name}")
                        
                        # Create custom pipeline
                        pipe_kwargs = {
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                            "max_new_tokens": generation_length,
                            "temperature": 0.1,
                            "batch_size": 1,
                            "do_sample": False,
                            "repetition_penalty": 1.1,
                            "pad_token_id": self.tokenizer.eos_token_id,
                            "return_full_text": False,
                            "eos_token_id": self.tokenizer.eos_token_id,
                        }
                        
                        temp_pipeline = pipeline("text-generation", **pipe_kwargs)
                        temp_llm = HuggingFacePipeline(pipeline=temp_pipeline)
                        
                        # Set up the setter function for this path
                        if path_name == 'combine_documents_chain.llm_chain.llm':
                            llm_setter = lambda llm: setattr(self.qa_chain.combine_documents_chain.llm_chain, 'llm', llm)
                        elif path_name == 'combine_documents_chain.llm':
                            llm_setter = lambda llm: setattr(self.qa_chain.combine_documents_chain, 'llm', llm)
                        elif path_name == 'llm_chain.llm':
                            llm_setter = lambda llm: setattr(self.qa_chain.llm_chain, 'llm', llm)
                        elif path_name == 'llm':
                            llm_setter = lambda llm: setattr(self.qa_chain, 'llm', llm)
                        
                        # Replace the LLM temporarily
                        llm_setter(temp_llm)
                        custom_pipeline_used = True
                        print(f"🔧 Using custom generation length: {generation_length} tokens")
                        break
                        
                except (AttributeError, TypeError):
                    continue  # Try next path
            
            if not custom_pipeline_used:
                print("⚠️ Could not access LLM in chain, using default generation length")
            
            # Execute query
            result = self.qa_chain({"query": optimized_question})
            
            # Restore original LLM if we modified it
            if custom_pipeline_used and original_llm is not None and llm_setter is not None:
                llm_setter(original_llm)
                
        except Exception as e:
            print(f"⚠️ Error with custom generation length: {e}")
            
            # Restore original state if something went wrong
            if custom_pipeline_used and original_llm is not None and llm_setter is not None:
                try:
                    llm_setter(original_llm)
                except:
                    pass  # If restoration fails, continue with fallback
            
            # Fallback to standard execution
            print("🔄 Falling back to standard execution with default generation length")
            try:
                result = self.qa_chain({"query": optimized_question})
            except Exception as fallback_error:
                print(f"❌ Even fallback failed: {fallback_error}")
                return {
                    "question": question,
                    "answer": "",
                    "error": f"Query execution failed: {fallback_error}",
                    "response_type": response_type,
                    "generation_method": "failed",
                    "generation_length": generation_length,
                    "custom_length_used": False,
                    "query_time": time.time() - start_time
                }
        
        query_time = time.time() - start_time
        source_docs = result.get("source_documents", [])
        raw_answer = result.get("result", "")
        
        # Handle JSON extraction for JSON responses
        parsed_json = None
        is_valid_json = False
        if response_type == "json":
            parsed_json = self._extract_json_from_response(raw_answer)
            is_valid_json = parsed_json is not None
            if is_valid_json:
                print("✅ Successfully extracted JSON from standard response")
            else:
                print("⚠️ Could not extract valid JSON from standard response")
        
        return {
            "question": question,
            "answer": raw_answer,
            "parsed_json": parsed_json,
            "is_valid_json": is_valid_json,
            "response_type": response_type,
            "generation_method": "standard",
            "generation_length": generation_length,
            "custom_length_used": custom_pipeline_used,
            "source_documents": self._format_source_docs(source_docs),
            "num_sources_used": len(source_docs),
            "query_time": query_time,
            "reranking_used": self.use_reranker and self.reranker is not None,
            "query_expansion_used": self.use_query_expansion and self.query_expander is not None
        }

    def _get_relevant_documents(self, question: str) -> List[Document]:
        """Get relevant documents using enhanced retrieval"""
        if self.use_reranker and self.reranker:
            if self.use_query_expansion and self.query_expander:
                # Use query expansion with reranking
                expanded_queries = self.query_expander.expand_query(question)
                all_docs = []
                for query in expanded_queries:
                    docs = self.vector_store.similarity_search(query, k=self.retrieval_k // len(expanded_queries))
                    all_docs.extend(docs)
                
                # Remove duplicates and rerank
                unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
                reranked_docs = self.reranker.rerank_documents(question, unique_docs, self.final_k)
                return [doc for doc, score in reranked_docs]
            else:
                # Standard reranking
                docs = self.vector_store.similarity_search(question, k=self.retrieval_k)
                reranked_docs = self.reranker.rerank_documents(question, docs, self.final_k)
                return [doc for doc, score in reranked_docs]
        else:
            # Standard similarity search
            return self.vector_store.similarity_search(question, k=self.final_k)

    def _format_source_docs(self, docs: List[Document]) -> List[Dict]:
        """Format source documents for response"""
        return [
            {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get('source', 'Unknown'),
                "rerank_score": doc.metadata.get('rerank_score', None),
                "chunk_method": doc.metadata.get('chunk_method', 'unknown')
            }
            for doc in docs
        ]

    def _optimize_query_for_model(self, question: str) -> str:
        """
        Optimize queries for faster processing by large models.
        Shortens prompts while preserving essential information.
        """
        # Check if we're using a large model
        large_models = ["7B", "13B", "30B", "70B", "8B", "14B", "32B", "65B"]
        is_large_model = any(size in str(self.llm) for size in large_models) if self.llm else False
        
        if not is_large_model or len(question) < 500:
            return question  # No optimization needed for small models or short queries
        
        # For large models, create a more concise prompt
        if "analysis" in question.lower() and "detailed" in question.lower():
            # Extract the essential parts for detailed analysis
            lines = question.split('\n')
            essential_lines = []
            
            for line in lines:
                # Keep important lines, skip verbose instructions
                if any(keyword in line.lower() for keyword in [
                    'entities:', 'relationships:', 'analyze', 'detail', 
                    'entity ', 'relationship ', 'structure', 'network'
                ]):
                    essential_lines.append(line)
                elif line.strip().startswith(('Entity ', 'Relationship ')):
                    essential_lines.append(line)
            
            if essential_lines:
                # Create a shorter, focused prompt
                optimized = "Perform detailed analysis:\n" + '\n'.join(essential_lines[:20])  # Limit to 20 lines
                optimized += "\n\nProvide JSON with entities and relationships structure."
                return optimized
        
        # General optimization: truncate very long queries
        if len(question) > 1000:
            return question[:1000] + "\n\n[Truncated for performance]"
        
        return question
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load documents based on file extension with better error handling"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return []
            
        extension = file_path.suffix.lower()
        
        try:
            print(f"📖 Loading {file_path.name}...")
            
            if extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif extension == '.csv':
                loader = CSVLoader(str(file_path))
            elif extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif extension == '.json':
                def json_metadata_func(record: dict, metadata: dict) -> dict:
                    metadata["source"] = str(file_path)
                    return metadata
                
                loader = JSONLoader(
                    file_path=str(file_path),
                    jq_schema='.[]' if isinstance(json.load(open(file_path, 'r')), list) else '.',
                    text_content=False,
                    metadata_func=json_metadata_func
                )
            else:
                print(f"❌ Unsupported file type: {extension}")
                return []
            
            documents = loader.load()
            
            # Add source metadata if missing
            for doc in documents:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = str(file_path)
                doc.metadata['file_type'] = extension
                doc.metadata['file_size'] = file_path.stat().st_size
            
            print(f"✅ Loaded {len(documents)} documents from {file_path.name}")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            print(f"📝 Full error: {traceback.format_exc()}")
            return []
    
    def add_documents_to_vector_store(self, documents: List[Document]):
        """Add documents to vector store with enhanced error handling and fallback"""
        if not documents:
            print("⚠️ No documents to add")
            return False
    
        try:
            # Check for existing documents to avoid duplicates
            documents_to_add = self._check_existing_documents(documents)
            if not documents_to_add:
                print("ℹ️ No new documents to add (all documents already exist)")
                if self.vector_store and not self.qa_chain:
                    self._initialize_qa_chain()
                return True
    
            if self.vector_store is None:
                print("🔨 Creating new vector store...")
                if self.use_milvus:
                    self.vector_store = Milvus.from_documents(
                        documents_to_add,
                        self.embeddings,
                        connection_args={
                            "uri": self.milvus_uri,
                            "user": self.milvus_user,
                            "password": self.milvus_password,
                            "secure": True
                        },
                        collection_name=self.collection_name
                    )
                    print(f"✅ Created Milvus vector store with {len(documents_to_add)} documents")
                else:
                    self.vector_store = FAISS.from_documents(documents_to_add, self.embeddings)
                    print(f"✅ Created FAISS vector store with {len(documents_to_add)} documents")
                self._initialize_qa_chain()
            else:
                print(f"📝 Adding {len(documents_to_add)} new documents to existing vector store...")
                texts = [doc.page_content for doc in documents_to_add]
                metadatas = [json.dumps(doc.metadata) for doc in documents_to_add]
                embeddings = self.embeddings.embed_documents(texts)
    
                if self.use_milvus:
                    try:
                        collection = Collection(self.collection_name)
                        collection.load()
                        schema = collection.schema
                        auto_id = schema.auto_id
    
                        if not auto_id:
                            print("⚠️ Collection requires manual IDs. Generating IDs...")
                            import uuid
                            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
                            insert_data = [ids, embeddings, texts, metadatas]
                        else:
                            insert_data = [embeddings, texts, metadatas]
    
                        # Ensure insert_data matches schema
                        expected_fields = len([f for f in schema.fields if f.name != 'id'])
                        if len(insert_data) != expected_fields:
                            raise ValueError(f"Data mismatch: expected {expected_fields} fields, got {len(insert_data)}")
    
                        self.vector_store.col.insert(insert_data)
                        print(f"✅ Successfully added {len(documents_to_add)} documents")
                    except Exception as milvus_error:
                        print(f"❌ Milvus error: {milvus_error}")
                        print(f"📝 Full error: {traceback.format_exc()}")
                        raise
                else:
                    self.vector_store.add_texts(texts=texts, metadatas=[json.loads(m) for m in metadatas])
                    print(f"✅ Successfully added {len(documents_to_add)} documents")
    
                self._initialize_qa_chain()
    
            return True
        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            print(f"📝 Full error: {traceback.format_exc()}")
            if self.use_milvus:
                print("🔄 Attempting fallback to FAISS...")
                try:
                    self.vector_store = FAISS.from_documents(documents_to_add, self.embeddings)
                    self.use_milvus = False
                    self._initialize_qa_chain()
                    print(f"✅ Fallback successful - created FAISS vector store with {len(documents_to_add)} documents")
                    return True
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
                    print(f"📝 Fallback error: {traceback.format_exc()}")
                    return False
            return False
    
    def ingest_file(self, file_path: str) -> bool:
        """Complete pipeline to ingest a file with enhanced features"""
        print(f"\n🔄 Ingesting file: {file_path}")
        
        # Load documents
        documents = self.load_document(file_path)
        if not documents:
            print(f"❌ Failed to load: {file_path}")
            return False
        
        # Process documents (split into chunks)
        processed_docs = self.process_documents(documents)
        if not processed_docs:
            print(f"❌ Failed to process: {file_path}")
            return False
        
        # Add to vector store
        success = self.add_documents_to_vector_store(processed_docs)
        if success:
            print(f"✅ Successfully ingested: {file_path}")
        else:
            print(f"❌ Failed to ingest: {file_path}")
            
        return success
    
    def ingest_files_parallel(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, bool]:
        """Ingest multiple files in parallel for better performance"""
        results = {}
        
        def ingest_single_file(file_path: str) -> Tuple[str, bool]:
            return file_path, self.ingest_file(file_path)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(ingest_single_file, fp): fp for fp in file_paths}
            
            for future in future_to_file:
                file_path, success = future.result()
                results[file_path] = success
        
        successful = sum(results.values())
        print(f"✅ Parallel ingestion complete: {successful}/{len(file_paths)} files successful")
        return results

    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Enhanced search with reranking support"""
        if not self.vector_store:
            return []
        
        try:
            if self.use_reranker and self.reranker:
                # Use parallel search and rerank for better performance
                return self.parallel_search_and_rerank(query, k)
            else:
                # Standard similarity search
                docs = self.vector_store.similarity_search(query, k=k)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": getattr(doc, 'similarity_score', None),
                        "source": doc.metadata.get('source', 'Unknown')
                    }
                    for doc in docs
                ]
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics about the vector store"""
        try:
            if self.use_milvus and self.vector_store:
                # Get Milvus collection stats
                collection = Collection(self.collection_name)
                collection.load()
                stats = {
                    "vector_store_type": "Milvus",
                    "collection_name": self.collection_name,
                    "total_entities": collection.num_entities,
                    "is_loaded": collection.is_empty == False,
                    "reranking_enabled": self.use_reranker and self.reranker is not None,
                    "reranker_model": self.reranker.model_name if self.reranker else None,
                    "retrieval_k": self.retrieval_k,
                    "final_k": self.final_k
                }
            elif self.vector_store:
                # FAISS stats
                stats = {
                    "vector_store_type": "FAISS",
                    "total_vectors": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0,
                    "reranking_enabled": self.use_reranker and self.reranker is not None,
                    "reranker_model": self.reranker.model_name if self.reranker else None,
                    "retrieval_k": self.retrieval_k,
                    "final_k": self.final_k
                }
            else:
                stats = {
                    "vector_store_type": "None", 
                    "total_entities": 0,
                    "reranking_enabled": False,
                    "reranker_model": None
                }
                
            return stats
            
        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}
    
    def inspect_collection(self):
        """Inspect the Milvus collection's schema and data with enhanced information"""
        if not self.use_milvus or not utility.has_collection(self.collection_name):
            print("⚠️ No Milvus collection or not using Milvus.")
            return None
    
        try:
            collection = Collection(self.collection_name)
            collection.load()
            schema = collection.schema
            print(f"🔍 Collection schema: {schema}")
            print(f"🔍 Auto ID: {schema.auto_id}")
            print(f"🔍 Total entities: {collection.num_entities}")
            print(f"🔍 Reranking enabled: {self.use_reranker and self.reranker is not None}")
            if self.reranker:
                print(f"🔍 Reranker model: {self.reranker.model_name}")
    
            # Query a sample of the data
            results = collection.query(expr="id >= 0", limit=5, output_fields=["text", "metadata"])
            print("🔍 Sample data (first 5 entities):")
            for result in results:
                metadata = json.loads(result['metadata']) if isinstance(result['metadata'], str) else result['metadata']
                print(f"ID: {result['id']}")
                print(f"Text preview: {result['text'][:100]}...")
                print(f"Source: {metadata.get('source', 'Unknown')}")
                print(f"Content hash: {metadata.get('content_hash', 'N/A')}")
                print("---")
            return results
        except Exception as e:
            print(f"❌ Error inspecting collection: {e}")
            return None
    
    def save_vector_store_local(self, path: str):
        """Save FAISS vector store locally for backup"""
        if not self.vector_store or self.use_milvus:
            print("⚠️ Only FAISS vector stores can be saved locally")
            return False
        
        try:
            self.vector_store.save_local(path)
            print(f"✅ Vector store saved to {path}")
            return True
        except Exception as e:
            print(f"❌ Error saving vector store: {e}")
            return False
    
    def load_vector_store_local(self, path: str):
        """Load FAISS vector store from local backup"""
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings)
            self.use_milvus = False
            self._initialize_qa_chain()
            print(f"✅ Vector store loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return False
    
    def benchmark_query_performance(self, test_queries: List[str], num_runs: int = 3) -> Dict[str, Any]:
        """Benchmark query performance with and without reranking"""
        if not test_queries:
            return {"error": "No test queries provided"}
        
        results = {
            "test_queries": test_queries,
            "num_runs": num_runs,
            "with_reranking": [],
            "without_reranking": [],
            "average_improvement": 0
        }
        
        # Test with reranking
        original_use_reranker = self.use_reranker
        self.use_reranker = True
        
        for query in test_queries:
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.query(query)
                times.append(time.time() - start_time)
            results["with_reranking"].append({
                "query": query,
                "times": times,
                "average": sum(times) / len(times)
            })
        
        # Test without reranking
        self.use_reranker = False
        self._initialize_qa_chain()  # Reinitialize without reranking
        
        for query in test_queries:
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.query(query)
                times.append(time.time() - start_time)
            results["without_reranking"].append({
                "query": query,
                "times": times,
                "average": sum(times) / len(times)
            })
        
        # Restore original setting
        self.use_reranker = original_use_reranker
        self._initialize_qa_chain()
        
        # Calculate improvements
        with_avg = np.mean([r["average"] for r in results["with_reranking"]])
        without_avg = np.mean([r["average"] for r in results["without_reranking"]])
        results["average_improvement"] = ((without_avg - with_avg) / without_avg * 100) if without_avg > 0 else 0
        
        print(f"📊 Benchmark complete:")
        print(f"   With reranking: {with_avg:.2f}s average")
        print(f"   Without reranking: {without_avg:.2f}s average")
        print(f"   Performance improvement: {results['average_improvement']:.1f}%")
        
        return results

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, handling various formats"""
        try:
            # First, try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON within the response
        json_patterns = [
            r'\{.*\}',  # Match from first { to last }
            r'```json\s*(\{.*?\})\s*```',  # Match JSON in code blocks
            r'```\s*(\{.*?\})\s*```',  # Match JSON in generic code blocks
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def validate_json_schema(self, json_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON response against a schema"""
        try:
            import jsonschema
            jsonschema.validate(instance=json_data, schema=schema)
            return {"valid": True, "errors": []}
        except ImportError:
            return {"valid": None, "errors": ["jsonschema not installed - cannot validate"]}
        except jsonschema.ValidationError as e:
            return {"valid": False, "errors": [str(e)]}
        except Exception as e:
            return {"valid": False, "errors": [f"Validation failed: {e}"]}

    def benchmark_response_types(self, test_queries: List[str], num_runs: int = 2) -> Dict[str, Any]:
        """Benchmark different response types and generation methods"""
        if not test_queries:
            return {"error": "No test queries provided"}
        
        results = {
            "test_queries": test_queries,
            "num_runs": num_runs,
            "conversational": [],
            "analytical": [],
            "json_standard": [],
            "json_controlled": [],
            "performance_summary": {}
        }
        
        response_types = ["conversational", "analytical", "json", "json"]
        controlled_flags = [False, False, False, True]
        result_keys = ["conversational", "analytical", "json_standard", "json_controlled"]
        
        for i, (resp_type, use_controlled) in enumerate(zip(response_types, controlled_flags)):
            result_key = result_keys[i]
            print(f"📊 Benchmarking {result_key}...")
            
            for query in test_queries:
                times = []
                success_count = 0
                
                for run in range(num_runs):
                    start_time = time.time()
                    try:
                        if resp_type == "json":
                            response = self.query(
                                query, 
                                response_type="json",
                                use_controlled_generation=use_controlled
                            )
                            # Check if JSON is valid
                            if response.get("is_valid_json", False):
                                success_count += 1
                        else:
                            response = self.query(query, response_type=resp_type)
                            if not response.get("error"):
                                success_count += 1
                        
                        elapsed_time = time.time() - start_time
                        times.append(elapsed_time)
                        
                    except Exception as e:
                        print(f"⚠️ Benchmark error: {e}")
                        times.append(float('inf'))
                
                results[result_key].append({
                    "query": query,
                    "times": times,
                    "average_time": sum(times) / len(times) if times else 0,
                    "success_rate": success_count / num_runs if num_runs > 0 else 0
                })
        
        # Calculate performance summary
        for key in result_keys:
            if results[key]:
                avg_time = np.mean([r["average_time"] for r in results[key]])
                avg_success = np.mean([r["success_rate"] for r in results[key]])
                results["performance_summary"][key] = {
                    "average_time": avg_time,
                    "average_success_rate": avg_success
                }
        
        print("📊 Benchmark complete!")
        return results

    def optimize_configuration(self, sample_queries: List[str] = None) -> Dict[str, Any]:
        """Automatically optimize configuration based on performance"""
        if not sample_queries:
            sample_queries = [
                "What are the main security vulnerabilities in this smart contract?",
                "Analyze the reentrancy risks in this code",
                "Return a JSON analysis of gas optimization opportunities"
            ]
        
        print("🔧 Starting configuration optimization...")
        
        # Test different configurations
        configs_to_test = [
            {"use_reranker": True, "use_query_expansion": True, "retrieval_k": 10},
            {"use_reranker": True, "use_query_expansion": False, "retrieval_k": 8},
            {"use_reranker": False, "use_query_expansion": True, "retrieval_k": 5},
            {"use_reranker": False, "use_query_expansion": False, "retrieval_k": 3},
        ]
        
        results = {}
        original_config = {
            "use_reranker": self.use_reranker,
            "use_query_expansion": self.use_query_expansion,
            "retrieval_k": self.retrieval_k
        }
        
        for i, config in enumerate(configs_to_test):
            print(f"🧪 Testing configuration {i+1}/{len(configs_to_test)}: {config}")
            
            # Apply configuration
            self.use_reranker = config["use_reranker"]
            self.use_query_expansion = config["use_query_expansion"]
            self.retrieval_k = config["retrieval_k"]
            
            # Reinitialize QA chain with new config
            self._initialize_qa_chain()
            
            # Test performance
            total_time = 0
            success_count = 0
            
            for query in sample_queries:
                try:
                    start_time = time.time()
                    result = self.query(query)
                    total_time += time.time() - start_time
                    
                    if not result.get("error"):
                        success_count += 1
                except Exception as e:
                    print(f"⚠️ Query failed: {e}")
            
            results[f"config_{i+1}"] = {
                "config": config,
                "average_time": total_time / len(sample_queries),
                "success_rate": success_count / len(sample_queries),
                "total_time": total_time
            }
        
        # Restore original configuration
        self.use_reranker = original_config["use_reranker"]
        self.use_query_expansion = original_config["use_query_expansion"]
        self.retrieval_k = original_config["retrieval_k"]
        self._initialize_qa_chain()
        
        # Find best configuration
        best_config = min(results.values(), key=lambda x: x["average_time"] * (1 - x["success_rate"]))
        
        optimization_result = {
            "tested_configurations": results,
            "recommended_config": best_config["config"],
            "improvement_potential": {
                "time_savings": (results["config_1"]["average_time"] - best_config["average_time"]) / results["config_1"]["average_time"] * 100,
                "current_config": original_config,
                "recommended_config": best_config["config"]
            }
        }
        
        print(f"✅ Optimization complete! Recommended config: {best_config['config']}")
        return optimization_result

    def optimize_gpu_configuration(self) -> Dict[str, Any]:
        """Optimize GPU configuration for better performance"""
        if not torch.cuda.is_available():
            return {"error": "No CUDA GPUs available"}
        
        optimization_result = {
            "current_config": {
                "multi_gpu_enabled": self.use_multi_gpu,
                "num_gpus": self.num_gpus,
                "device_map": str(self.device_map)
            },
            "recommendations": [],
            "memory_optimization": {},
            "performance_tips": []
        }
        
        # Memory optimization recommendations
        gpu_stats = self._get_memory_usage_per_gpu()
        for gpu_id, stats in gpu_stats.items():
            if stats["utilization_percent"] > 90:
                optimization_result["recommendations"].append(
                    f"GPU {gpu_id} is at {stats['utilization_percent']:.1f}% usage - consider quantization"
                )
            elif stats["utilization_percent"] < 50:
                optimization_result["recommendations"].append(
                    f"GPU {gpu_id} is underutilized at {stats['utilization_percent']:.1f}% - consider larger batch sizes"
                )
        
        # Multi-GPU recommendations
        if self.num_gpus > 1 and not self.use_multi_gpu:
            optimization_result["recommendations"].append(
                f"You have {self.num_gpus} GPUs available but multi-GPU is disabled. Enable with use_multi_gpu=True"
            )
        
        # Performance tips
        optimization_result["performance_tips"] = [
            "Use quantization (4-bit or 8-bit) to reduce memory usage",
            "Enable multi-GPU for models larger than single GPU memory",
            "Consider flash-attention for faster inference",
            "Use tensor parallelism for very large models",
            "Monitor GPU utilization and adjust batch sizes accordingly"
        ]
        
        return optimization_result

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            print("🧹 Clearing GPU memory cache...")
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("✅ GPU memory cache cleared")
            self._display_gpu_memory_usage()
        else:
            print("⚠️ No CUDA GPUs available")

    def switch_to_single_gpu(self, gpu_id: int = 0):
        """Switch from multi-GPU to single GPU mode"""
        if not torch.cuda.is_available():
            print("⚠️ No CUDA GPUs available")
            return False
        
        if gpu_id >= self.num_gpus:
            print(f"❌ GPU {gpu_id} not available. Available GPUs: 0-{self.num_gpus-1}")
            return False
        
        print(f"🔄 Switching to single GPU mode (GPU {gpu_id})...")
        
        # Clear current GPU memory
        self.clear_gpu_memory()
        
        # Update configuration
        self.use_multi_gpu = False
        self.device_map = f"cuda:{gpu_id}"
        
        # Reinitialize model if needed
        if self.model is not None:
            print("🔄 Moving model to single GPU...")
            try:
                self.model = self.model.to(f"cuda:{gpu_id}")
                print(f"✅ Model moved to GPU {gpu_id}")
                return True
            except Exception as e:
                print(f"❌ Failed to move model: {e}")
                return False
        
        return True

    def enable_multi_gpu_if_available(self):
        """Enable multi-GPU mode if multiple GPUs are available"""
        if self.num_gpus > 1 and not self.use_multi_gpu:
            print(f"🖥️ Enabling multi-GPU mode with {self.num_gpus} GPUs...")
            self.use_multi_gpu = True
            self.device_map = self._create_device_map()
            
            # Note: Model reinitialization required for this to take effect
            print("⚠️ Note: You need to reinitialize the model for multi-GPU to take effect")
            print("   Call: rag._initialize_advanced_llm(model_name)")
            return True
        elif self.num_gpus <= 1:
            print("⚠️ Only 1 GPU available, cannot enable multi-GPU mode")
            return False
        else:
            print("ℹ️ Multi-GPU mode already enabled")
            return True

    def get_gpu_recommendations(self) -> Dict[str, Any]:
        """Get GPU optimization recommendations"""
        if not torch.cuda.is_available():
            return {"error": "No CUDA GPUs available"}
        
        recommendations = {
            "hardware_info": {},
            "memory_recommendations": [],
            "performance_recommendations": [],
            "model_size_recommendations": []
        }
        
        # Hardware info
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            recommendations["hardware_info"][f"gpu_{i}"] = {
                "name": props.name,
                "memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            }
        
        # Memory recommendations
        total_memory = sum(
            torch.cuda.get_device_properties(i).total_memory 
            for i in range(self.num_gpus)
        ) / (1024**3)
        
        if total_memory > 80:
            recommendations["model_size_recommendations"].append(
                "Your system can handle large models (70B+ parameters)"
            )
        elif total_memory > 40:
            recommendations["model_size_recommendations"].append(
                "Your system can handle medium-large models (30-70B parameters)"
            )
        elif total_memory > 20:
            recommendations["model_size_recommendations"].append(
                "Your system can handle medium models (7-30B parameters)"
            )
        else:
            recommendations["model_size_recommendations"].append(
                "Consider smaller models (<7B parameters) or use quantization"
            )
        
        # Performance recommendations
        if self.num_gpus > 1:
            recommendations["performance_recommendations"].extend([
                "Enable multi-GPU with use_multi_gpu=True for better performance",
                "Use tensor parallelism for very large models",
                "Consider pipeline parallelism for extremely large models"
            ])
        
        if any(torch.cuda.get_device_properties(i).major >= 8 for i in range(self.num_gpus)):
            recommendations["performance_recommendations"].append(
                "Your GPUs support flash-attention - install flash-attn for faster inference"
            )
        
        return recommendations

    def _ensure_device_consistency(self):
        """Ensure all components are on consistent devices"""
        if not torch.cuda.is_available():
            return
        
        try:
            # Get model device
            if self.model is not None:
                model_device = next(self.model.parameters()).device
                print(f"🔧 Model device: {model_device}")
                
                # Ensure embeddings are on compatible device
                if hasattr(self.embeddings, 'client') and hasattr(self.embeddings.client, 'device'):
                    print(f"🔧 Embeddings device: {self.embeddings.client.device}")
                
                # Clear any cached tensors that might be on wrong device
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"⚠️ Device consistency check failed: {e}")

    def fix_device_issues(self):
        """Fix common device mismatch issues"""
        print("🔧 Fixing device consistency issues...")
        
        if not torch.cuda.is_available():
            print("💻 Running on CPU - no device issues to fix")
            return
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Check device consistency
            self._ensure_device_consistency()
            
            print("✅ Device issues fixed")
            
        except Exception as e:
            print(f"⚠️ Could not fix device issues: {e}")
            print("💡 Try restarting the RAG system if device issues persist")

    def direct_generation_with_length(self, question: str, generation_length: int = 1024) -> Dict[str, Any]:
        """
        Direct generation with custom length, bypassing QA chain issues
        
        Args:
            question: The question to answer
            generation_length: Maximum tokens to generate
            
        Returns:
            Dict with generation result
        """
        try:
            print(f"🔧 Direct generation with {generation_length} tokens")
            
            # Get relevant documents
            docs = self._get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = f"""Based on the provided context, answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_length,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):].strip()
            
            return {
                "question": question,
                "answer": answer,
                "generation_method": "direct",
                "generation_length": generation_length,
                "source_documents": self._format_source_docs(docs),
                "num_sources_used": len(docs),
                "success": True
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": "",
                "generation_method": "direct",
                "generation_length": generation_length,
                "error": str(e),
                "success": False
            }

    def create_custom_schema(self, schema_type: str = "default") -> Dict[str, Any]:
        """Create predefined JSON schemas for common use cases"""
        schemas = {
            "default": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 100}
                },
                "required": ["summary", "key_points"]
            },
            
            "analysis": {
                "type": "object",
                "properties": {
                    "main_findings": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "importance": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                            },
                            "required": ["name", "description"]
                        }
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "overall_assessment": {"type": "string"}
                },
                "required": ["main_findings", "overall_assessment"]
            },
            
            "detailed": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "description": {"type": "string"},
                                "attributes": {"type": "array", "items": {"type": "string"}},
                                "relationships": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "target_id": {"type": "string"},
                                            "type": {"type": "string"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["target_id", "type"]
                                    }
                                }
                            },
                            "required": ["id", "description"]
                        }
                    },
                    "summary": {"type": "string"}
                },
                "required": ["entities", "summary"]
            }
        }
        
        return schemas.get(schema_type, schemas["default"])

    def get_enhanced_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced vector store"""
        try:
            base_stats = self.get_collection_stats()
            
            enhanced_stats = {
                **base_stats,
                "advanced_features": {
                    "controlled_generation": self.use_controlled_generation,
                    "semantic_chunking": self.use_semantic_chunking,
                    "query_expansion": self.use_query_expansion,
                    "reranking": self.use_reranker,
                    "multi_gpu": self.use_multi_gpu,
                },
                "gpu_configuration": {
                    "cuda_available": torch.cuda.is_available(),
                    "num_gpus": self.num_gpus,
                    "gpu_devices": self.gpu_devices,
                    "multi_gpu_enabled": self.use_multi_gpu and self.num_gpus > 1,
                    "device_map": str(self.device_map) if self.device_map else None,
                    "memory_stats": self._get_memory_usage_per_gpu() if torch.cuda.is_available() else {}
                },
                "configurations": {
                    "chunking_strategy": self.chunking_config.strategy if self.use_semantic_chunking else "recursive",
                    "chunk_size": self.chunking_config.chunk_size,
                    "semantic_threshold": self.chunking_config.semantic_threshold if self.use_semantic_chunking else None,
                    "query_expansion_config": {
                        "use_synonyms": self.query_expansion_config.use_synonyms,
                        "max_expansions": self.query_expansion_config.max_expansions
                    } if self.use_query_expansion else None,
                    "retrieval_k": self.retrieval_k,
                    "final_k": self.final_k
                },
                "model_info": {
                    "controlled_generator_available": self.controlled_generator is not None,
                    "guidance_available": GUIDANCE_AVAILABLE,
                    "quantization": self.quantization_bits
                }
            }
            
            return enhanced_stats
            
        except Exception as e:
            return {"error": f"Failed to get enhanced stats: {e}"}

    def parallel_search_and_rerank(self, query: str, k: int = 5) -> List[Dict]:
        """
        Parallel search and reranking for even faster results
        
        Args:
            query: Search query
            k: Number of final results to return
            
        Returns:
            List of reranked documents with scores
        """
        if not self.vector_store:
            return []
        
        try:
            start_time = time.time()
            
            # Use ThreadPoolExecutor for parallel operations
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit vector search
                search_future = executor.submit(
                    self.vector_store.similarity_search, 
                    query, 
                    self.retrieval_k
                )
                
                # Get search results
                docs = search_future.result()
                
                if self.use_reranker and self.reranker and docs:
                    # Submit reranking
                    rerank_future = executor.submit(
                        self.reranker.rerank_documents,
                        query,
                        docs,
                        k
                    )
                    
                    # Get reranked results
                    reranked_docs_with_scores = rerank_future.result()
                else:
                    reranked_docs_with_scores = [(doc, 0.0) for doc in docs[:k]]
            
            search_time = time.time() - start_time
            print(f"⚡ Parallel search and rerank completed in {search_time:.2f} seconds")
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "rerank_score": float(score),
                    "source": doc.metadata.get('source', 'Unknown')
                }
                for doc, score in reranked_docs_with_scores
            ]
            
        except Exception as e:
            print(f"❌ Error in parallel search and rerank: {e}")
            return []

    # Convenience methods for common query patterns
    def ask_json(self, question: str, schema: Optional[Dict] = None, generation_length: int = 1024) -> Dict[str, Any]:
        """Convenience method for JSON queries"""
        return self.query(
            question=question,
            response_type="json",
            schema=schema,
            generation_length=generation_length
        )
    
    def ask_analytical(self, question: str, generation_length: int = 3072) -> Dict[str, Any]:
        """Convenience method for analytical queries"""
        return self.query(
            question=question,
            response_type="analytical",
            generation_length=generation_length
        )
    
    def ask_conversational(self, question: str, generation_length: int = 2048) -> Dict[str, Any]:
        """Convenience method for conversational queries"""
        return self.query(
            question=question,
            response_type="conversational",
            generation_length=generation_length
        )
    
    def ask_streaming(self, question: str, callback_func=None, generation_length: int = 2048) -> Iterator[str]:
        """Convenience method for streaming queries"""
        return self.query(
            question=question,
            use_streaming=True,
            callback_func=callback_func,
            generation_length=generation_length
        )
    
    def ask_detailed_analysis(self, question: str, generation_length: int = 8192) -> Dict[str, Any]:
        """Convenience method for detailed analysis with custom schema"""
        schema = self.create_custom_schema("detailed")
        return self.query(
            question=question,
            response_type="json",
            schema=schema,
            generation_length=generation_length,
            use_direct_generation=True  # Use direct generation for better custom length support
        )

def example_usage():
    """Example usage of the RAG system with different query types"""
    import os
    from pathlib import Path
    
    # Initialize the RAG system
    rag = RAGSystem(
        use_milvus=False,  # Use FAISS for local vector store
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for example
        llm_model="Qwen/Qwen2.5-1.5B-Instruct",  # Smaller model for example
        use_reranker=True,
        use_controlled_generation=True,
        use_semantic_chunking=True
    )
    
    # Create a sample directory for documents
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample documents
    sample_docs = {
        "programming.txt": """
# Python Programming Guide

Python is a high-level, interpreted programming language known for its readability and versatility.

## Key Features
- Easy to learn and use
- Extensive standard library
- Support for multiple programming paradigms
- Dynamic typing and memory management
- Strong community support

## Common Use Cases
- Web development
- Data analysis and visualization
- Machine learning
- Automation and scripting
- Scientific computing

## Basic Syntax
```python
# This is a comment
def hello_world():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()
```
        """,
        
        "data_science.txt": """
# Introduction to Data Science

Data science combines domain knowledge, programming skills, and statistical understanding to extract insights from data.

## Core Components
- Data collection and cleaning
- Exploratory data analysis
- Statistical modeling
- Machine learning algorithms
- Data visualization and communication

## Popular Libraries
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib and Seaborn: Visualization
- Scikit-learn: Machine learning
- TensorFlow and PyTorch: Deep learning

## Data Science Workflow
1. Define the problem
2. Collect and prepare data
3. Explore and analyze
4. Build and evaluate models
5. Deploy and monitor solutions
        """,
        
        "cloud_computing.txt": """
# Cloud Computing Overview

Cloud computing delivers computing services over the internet, offering flexible resources and economies of scale.

## Service Models
- Infrastructure as a Service (IaaS)
- Platform as a Service (PaaS)
- Software as a Service (SaaS)
- Function as a Service (FaaS)

## Deployment Models
- Public cloud
- Private cloud
- Hybrid cloud
- Multi-cloud

## Benefits
- Cost efficiency
- Scalability
- Reliability
- Global accessibility
- Reduced maintenance
        """
    }
    
    # Write sample documents to files
    for filename, content in sample_docs.items():
        with open(sample_dir / filename, "w") as f:
            f.write(content)
    
    # Ingest the sample documents
    for filename in sample_docs.keys():
        rag.ingest_file(str(sample_dir / filename))
    
    print("\n" + "="*50)
    print("RAG System Example Usage")
    print("="*50)
    
    # Example 1: Basic conversational query
    print("\n\n1. Basic Conversational Query:")
    print("-"*40)
    response = rag.ask_conversational("What are the key features of Python?")
    print(f"Response: {response['answer']}")
    
    # Example 2: Analytical query
    print("\n\n2. Analytical Query:")
    print("-"*40)
    response = rag.ask_analytical("Compare data science libraries and their purposes.")
    print(f"Response: {response['answer']}")
    
    # Example 3: JSON structured response
    print("\n\n3. JSON Structured Response:")
    print("-"*40)
    response = rag.ask_json("List all cloud computing service models with descriptions.")
    print(f"Structured JSON Response:")
    if response.get('parsed_json'):
        import json
        print(json.dumps(response['parsed_json'], indent=2))
    else:
        print(response['answer'])
    
    # Example 4: Custom schema for detailed analysis
    print("\n\n4. Detailed Analysis with Custom Schema:")
    print("-"*40)
    custom_schema = {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "importance": {"type": "string", "enum": ["low", "medium", "high"]}
                    }
                }
            },
            "summary": {"type": "string"}
        }
    }
    response = rag.query(
        question="Analyze programming, data science, and cloud computing topics in the documents.", 
        response_type="json",
        schema=custom_schema
    )
    print(f"Detailed Analysis Response:")
    if response.get('parsed_json'):
        import json
        print(json.dumps(response['parsed_json'], indent=2))
    else:
        print(response['answer'])
    
    # Example 5: Search for similar documents
    print("\n\n5. Similar Document Search:")
    print("-"*40)
    similar_docs = rag.search_similar_documents("machine learning applications", k=2)
    print(f"Found {len(similar_docs)} similar documents:")
    for i, doc in enumerate(similar_docs):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc['source']}")
        print(f"Content Preview: {doc['content'][:150]}...")
    
    # Clean up sample files
    print("\n\nCleaning up example files...")
    for filename in sample_docs.keys():
        (sample_dir / filename).unlink(missing_ok=True)
    sample_dir.rmdir()
    
    print("\nExample completed!")

if __name__ == "__main__":
    example_usage()