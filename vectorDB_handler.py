import os
from typing import List, Dict, Any
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import MongodbLoader
from langchain.schema import Document
from pymongo import MongoClient
import pandas as pd

class ProcurementRAG:
    def __init__(
        self,
        mongo_uri: str,
        database: str,
        collection: str,
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the RAG pipeline for procurement data
        
        Args:
            mongo_uri (str): MongoDB connection URI
            database (str): MongoDB database name
            collection (str): MongoDB collection name
            persist_directory (str): Directory to persist ChromaDB
        """
        # MongoDB setup
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[database]
        self.collection = self.db[collection]
        
        # Initialize HuggingFace embeddings
        # Using all-MiniLM-L6-v2 as it provides good performance and efficiency
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # ChromaDB setup
        self.persist_directory = persist_directory
        self.vector_store = None

    def _format_procurement_record(self, record: Dict[str, Any]) -> str:
        """
        Format a procurement record into a meaningful text representation
        
        Args:
            record (Dict): MongoDB document
            
        Returns:
            str: Formatted text representation
        """
        # Remove None values and format dates
        record = {k: v for k, v in record.items() if v is not None}
        
        # Format creation and purchase dates if they exist
        creation_date = record.get('Creation Date', '')
        if creation_date:
            try:
                creation_date = pd.to_datetime(creation_date).strftime('%Y-%m-%d')
            except:
                pass
                
        purchase_date = record.get('Purchase Date', '')
        if purchase_date:
            try:
                purchase_date = pd.to_datetime(purchase_date).strftime('%Y-%m-%d')
            except:
                pass

        # Create a structured text representation
        text_parts = [
            f"Purchase Order: {record.get('Purchase Order Number', 'N/A')}",
            f"Department: {record.get('Department Name', 'N/A')}",
            f"Supplier: {record.get('Supplier Name', 'N/A')}",
            f"Creation Date: {creation_date}",
            f"Purchase Date: {purchase_date}",
            f"Fiscal Year: {record.get('Fiscal Year', 'N/A')}",
            f"Item: {record.get('Item Name', 'N/A')}",
            f"Description: {record.get('Item Description', 'N/A')}",
            f"Quantity: {record.get('Quantity', 'N/A')}",
            f"Unit Price: ${record.get('Unit Price', 0):.2f}",
            f"Total Price: ${record.get('Total Price', 0):.2f}",
            f"Acquisition Type: {record.get('Acquisition Type', 'N/A')}",
            f"Acquisition Method: {record.get('Acquisition Method', 'N/A')}"
        ]
        
        # Add optional fields if they exist
        if record.get('Supplier Qualifications'):
            text_parts.append(f"Supplier Qualifications: {record['Supplier Qualifications']}")
        if record.get('Classification Codes'):
            text_parts.append(f"Classification Codes: {record['Classification Codes']}")
        
        # Join all parts with newlines
        return "\n".join(text_parts)

    def _create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for a procurement record
        
        Args:
            record (Dict): MongoDB document
            
        Returns:
            Dict: Metadata dictionary
        """
        return {
            'purchase_order': str(record.get('Purchase Order Number', '')),
            'department': str(record.get('Department Name', '')),
            'supplier': str(record.get('Supplier Name', '')),
            'fiscal_year': str(record.get('Fiscal Year', '')),
            'total_price': float(record.get('Total Price', 0)),
            'creation_date': str(record.get('Creation Date', '')),
            'acquisition_type': str(record.get('Acquisition Type', ''))
        }

    def load_data_to_vectorstore(self, batch_size: int = 1000) -> None:
        """
        Load data from MongoDB to ChromaDB in batches
        
        Args:
            batch_size (int): Number of records to process in each batch
        """
        try:
            total_documents = self.collection.count_documents({})
            print(f"Total documents to process: {total_documents}")
            
            # Initialize ChromaDB
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            processed = 0
            cursor = self.collection.find({})
            
            while True:
                # Process documents in batches
                batch_documents = []
                batch_metadatas = []
                batch_ids = []
                
                # Collect batch of documents
                for _ in range(batch_size):
                    try:
                        record = next(cursor)
                    except StopIteration:
                        break
                        
                    # Format the record and create metadata
                    formatted_text = self._format_procurement_record(record)
                    metadata = self._create_metadata(record)
                    
                    batch_documents.append(formatted_text)
                    batch_metadatas.append(metadata)
                    batch_ids.append(str(record['_id']))
                
                if not batch_documents:
                    break
                # Add documents to ChromaDB
                if processed > 189000:
                    self.vector_store.add_texts(
                        texts=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                
                processed += len(batch_documents)
                print(f"Processed {processed}/{total_documents} documents")
            
            # Persist the vector store
            self.vector_store.persist()
            print("Data loading completed successfully")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            filter_dict (Dict): Optional filters for metadata
            
        Returns:
            List[Dict]: List of matching documents with scores
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load data first.")
        
        try:
            # Perform the search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                }
                formatted_results.append(result)
                
            return formatted_results
            
        except Exception as e:
            print(f"Error performing search: {str(e)}")
            raise

    def get_filtered_results(
        self,
        query: str,
        department: str = None,
        fiscal_year: str = None,
        min_price: float = None,
        max_price: float = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get filtered search results
        
        Args:
            query (str): Search query
            department (str): Filter by department
            fiscal_year (str): Filter by fiscal year
            min_price (float): Minimum total price
            max_price (float): Maximum total price
            k (int): Number of results to return
            
        Returns:
            List[Dict]: Filtered search results
        """
        # Build filter dictionary
        filter_dict = {}
        
        if department:
            filter_dict['department'] = department
        if fiscal_year:
            filter_dict['fiscal_year'] = fiscal_year
        if min_price is not None:
            filter_dict['total_price'] = {'$gte': min_price}
        if max_price is not None:
            if 'total_price' in filter_dict:
                filter_dict['total_price']['$lte'] = max_price
            else:
                filter_dict['total_price'] = {'$lte': max_price}
        
        return self.similarity_search(query, k=k, filter_dict=filter_dict)

def main():
    # Configuration
    config = {
        'mongo_uri': 'mongodb://localhost:27017/',
        'database': 'procurement_db',
        'collection': 'purchases',
        'persist_directory': './chroma_db'
    }
    
    # Initialize RAG pipeline
    rag = ProcurementRAG(**config)
    
    # Load data to vector store
    rag.load_data_to_vectorstore()
    
    # Example searches
    print("\n=== Basic Search ===")
    results = rag.similarity_search("computer equipment purchases", k=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['content'][:200]}...")
        print(f"Score: {result['similarity_score']}")
        
    print("\n=== Filtered Search ===")
    filtered_results = rag.get_filtered_results(
        query="office supplies",
        department="Consumer Affairs, Department of",
        fiscal_year="2013-2014",
        min_price=100,
        k=3
    )
    for i, result in enumerate(filtered_results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['content'][:200]}...")
        print(f"Score: {result['similarity_score']}")

if __name__ == "__main__":
    main()