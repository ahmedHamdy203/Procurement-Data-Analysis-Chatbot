import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd 

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Utility imports
import torch
from huggingface_hub import hf_hub_download
import warnings

# Suppress warnings and tensorflow logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataclass
class ModelConfig:
    model_id: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_dir: str = "models"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    context_length: int = 1024
    gpu_layers: int = 0
    threads: int = 4
    batch_size: int = 4

@dataclass
class RAGConfig:
    persist_directory: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retriever_k: int = 2
    chunk_size: int = 300
    embedding_batch_size: int = 16

class Logger:
    @staticmethod
    def setup(name: str, log_file: str="rag_system.log") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler with UTF-8 encoding
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Clear existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
            
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = Logger.setup("ModelManager", "LLM_download_status.log")
        
    def download_model(self) -> str:
        """Download and verify the model"""
        try:
            os.makedirs(self.config.model_dir, exist_ok=True)
            local_path = os.path.join(self.config.model_dir, self.config.model_file)
            
            if not os.path.exists(local_path):
                self.logger.info(f"Downloading model {self.config.model_file}...")
                local_path = hf_hub_download(
                    repo_id=self.config.model_id,
                    filename=self.config.model_file,
                    local_dir=self.config.model_dir,
                    resume_download=True
                )
                self.logger.info("Model downloaded successfully!")
            else:
                self.logger.info("Using existing model file.")
            
            return local_path
            
        except Exception as e:
            self.logger.error(f"Error downloading model: {str(e)}")
            raise

class RAGPipeline:
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        rag_config: RAGConfig = RAGConfig()
    ):
        """Initialize the RAG pipeline"""
        self.model_config = model_config
        self.rag_config = rag_config
        self.logger = Logger.setup("RAGPipeline", "RAG_pipeline.log")
        
        # Initialize components
        self.setup_pipeline()

    def setup_pipeline(self):
        """Set up all pipeline components"""
        try:
            self.logger.info("Initializing RAG pipeline...")
            
            # Download model
            model_manager = ModelManager(self.model_config)
            self.model_path = model_manager.download_model()
            
            # Initialize components
            self.setup_embeddings()
            self.setup_vectorstore()
            self.setup_llm()
            self.setup_qa_chain()
            
            self.logger.info("RAG pipeline initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise

    def setup_embeddings(self):
        """Initialize the embedding model"""
        self.logger.info("Setting up embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.rag_config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': self.rag_config.embedding_batch_size
            }
        )

    def setup_vectorstore(self):
        """Initialize the vector store"""
        self.logger.info("Setting up vector store...")
        self.vectorstore = Chroma(
            persist_directory=self.rag_config.persist_directory,
            embedding_function=self.embeddings
        )

    def setup_llm(self):
        """Initialize the language model"""
        self.logger.info("Setting up LLM...")
        self.llm = CTransformers(
            model=self.model_path,
            model_type="llama",
            config={
                'max_new_tokens': self.model_config.max_new_tokens,
                'temperature': self.model_config.temperature,
                'top_p': self.model_config.top_p,
                'context_length': self.model_config.context_length,
                'gpu_layers': self.model_config.gpu_layers,
                'batch_size': self.model_config.batch_size,
                'threads': self.model_config.threads
            }
        )

    def setup_qa_chain(self):
        """Initialize the QA chain with custom prompt"""
        template = """Use the following pieces of context to answer the question about procurement data.
        Be specific and include numbers, dates, and facts from the context when available.
        If you cannot find the answer in the context, say "I don't have enough information."

        Context: {context}

        Question: {question}

        Answer: """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create retriever with simple similarity search
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.rag_config.retriever_k}
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def query(
        self,
        question: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Process a query with retries and timing
        """
        start_time = datetime.now()
        retries = 0
        
        while retries <= max_retries:
            try:
                self.logger.info(f"\nProcessing query: {question}")
                
                # Get relevant documents using direct similarity search
                docs = self.vectorstore.similarity_search(
                    question,
                    k=self.rag_config.retriever_k
                )
                
                # Log retrieved documents
                for i, doc in enumerate(docs):
                    self.logger.info(f"\nRetrieved Document {i+1}:")
                    self.logger.info(f"Content: {doc.page_content}")
                    self.logger.info(f"Metadata: {doc.metadata}")

                # Generate answer
                self.logger.info("Generating answer...")
                response = self.qa_chain({"query": question})
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    "answer": response["result"],
                    "source_documents": response["source_documents"],
                    "processing_time": processing_time
                }
                
                self.logger.info(f"Answer: {result['answer']}")
                self.logger.info(f"Processing time: {processing_time:.2f} seconds")
                
                return result
                
            except Exception as e:
                retries += 1
                self.logger.warning(f"Attempt {retries} failed: {str(e)}")
                if retries <= max_retries:
                    self.logger.info("Retrying...")
                else:
                    self.logger.error("Max retries reached")
                    return {
                        "error": str(e),
                        "processing_time": (datetime.now() - start_time).total_seconds()
                    }

def test_rag_pipeline():
    """Test function for the RAG pipeline"""
    try:
        print("\nInitializing RAG pipeline...")
        rag = RAGPipeline()
        
        test_questions = [            
            "Which department spent the most on computer equipment?",
            "What are the most common items purchased by the Consumer Affairs department?",
            "Can you provide an overview of the top spending departments and the types of items they typically purchase?",           
        ]
        examples = []
        for question in test_questions:
            print(f"\nQuestion: {question}")
            result = rag.query(question)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nAnswer: {result['answer']}")
                print(f"Processing time: {result['processing_time']:.2f} seconds")
                examples.append({"Question": question, "Answer": result["answer"]})
                print("\nSource Documents:")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"\nDocument {i}:")
                    print(f"Content: {doc.page_content[:200]}...")
            
            print("-" * 80)
        pd.DataFrame(examples).to_csv("./Example_QAs.csv")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    test_rag_pipeline()