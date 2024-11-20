import chromadb
from chromadb.config import Settings
from pymongo import MongoClient
from typing import Dict, Any, List
import logging
from datetime import datetime

class RecordChecker:
    def __init__(
        self,
        mongo_uri: str,
        database: str,
        collection: str,
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the record checker
        
        Args:
            mongo_uri (str): MongoDB connection URI
            database (str): MongoDB database name
            collection (str): MongoDB collection name
            persist_directory (str): ChromaDB persistence directory
        """
        # Setup logging
        self._setup_logging()
        
        # MongoDB setup
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[database]
        self.collection = self.db[collection]
        
        # ChromaDB setup
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('record_checker.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_records(self) -> Dict[str, Any]:
        """
        Check and compare records in both databases
        
        Returns:
            Dict: Statistics about record counts and discrepancies
        """
        stats = {
            'timestamp': datetime.now(),
            'mongodb_count': 0,
            'chromadb_count': 0,
            'difference': 0,
            'collections': [],
            'details': {}
        }
        
        try:
            # Get MongoDB count
            mongo_count = self.collection.count_documents({})
            stats['mongodb_count'] = mongo_count
            self.logger.info(f"MongoDB record count: {mongo_count}")
            
            # Get ChromaDB collections
            collections = self.chroma_client.list_collections()
            stats['collections'] = [col.name for col in collections]
            
            # Count records in each ChromaDB collection
            total_chroma_count = 0
            collection_counts = {}
            
            for collection in collections:
                count = collection.count()
                collection_counts[collection.name] = count
                total_chroma_count += count
            
            stats['chromadb_count'] = total_chroma_count
            stats['difference'] = mongo_count - total_chroma_count
            stats['details']['collection_counts'] = collection_counts
            
            self.logger.info(f"ChromaDB total record count: {total_chroma_count}")
            self.logger.info(f"Difference in records: {stats['difference']}")
            
            # Check for potential issues
            if stats['difference'] != 0:
                self.logger.warning(
                    f"Record count mismatch detected: "
                    f"MongoDB has {mongo_count} records, "
                    f"ChromaDB has {total_chroma_count} records"
                )
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error checking records: {str(e)}")
            raise

    def get_sample_records(self, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get sample records from both databases for comparison
        
        Args:
            limit (int): Number of records to sample
            
        Returns:
            Dict: Sample records from both databases
        """
        samples = {
            'mongodb_samples': [],
            'chromadb_samples': []
        }
        
        try:
            # Get MongoDB samples
            mongo_samples = list(self.collection.find().limit(limit))
            samples['mongodb_samples'] = mongo_samples
            
            # Get ChromaDB samples
            collections = self.chroma_client.list_collections()
            for collection in collections:
                results = collection.get(limit=limit)
                if results and 'metadatas' in results:
                    samples['chromadb_samples'].extend(results['metadatas'])
                
            return samples
            
        except Exception as e:
            self.logger.error(f"Error getting sample records: {str(e)}")
            raise

    def verify_record_integrity(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Verify the integrity of records between MongoDB and ChromaDB
        
        Args:
            sample_size (int): Number of records to check
            
        Returns:
            Dict: Verification results
        """
        results = {
            'timestamp': datetime.now(),
            'records_checked': 0,
            'matching_records': 0,
            'missing_records': 0,
            'mismatched_records': 0,
            'issues': []
        }
        
        try:
            # Get sample of MongoDB records
            mongo_records = list(self.collection.find().limit(sample_size))
            results['records_checked'] = len(mongo_records)
            
            # Check each record in ChromaDB
            collections = self.chroma_client.list_collections()
            
            for record in mongo_records:
                record_id = str(record['_id'])
                found = False
                
                # Search for record in ChromaDB collections
                for collection in collections:
                    try:
                        chroma_record = collection.get(ids=[record_id])
                        if chroma_record and chroma_record['metadatas']:
                            found = True
                            # Verify key fields match
                            if self._verify_record_match(record, chroma_record['metadatas'][0]):
                                results['matching_records'] += 1
                            else:
                                results['mismatched_records'] += 1
                                results['issues'].append({
                                    'record_id': record_id,
                                    'type': 'mismatch',
                                    'details': 'Record content differs between databases'
                                })
                            break
                    except Exception as e:
                        self.logger.warning(f"Error checking record {record_id}: {str(e)}")
                        
                if not found:
                    results['missing_records'] += 1
                    results['issues'].append({
                        'record_id': record_id,
                        'type': 'missing',
                        'details': 'Record exists in MongoDB but not in ChromaDB'
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error verifying record integrity: {str(e)}")
            raise

    def _verify_record_match(self, mongo_record: Dict, chroma_record: Dict) -> bool:
        """
        Verify key fields match between MongoDB and ChromaDB records
        
        Args:
            mongo_record (Dict): MongoDB record
            chroma_record (Dict): ChromaDB record
            
        Returns:
            bool: True if records match
        """
        key_fields = [
            'purchase_order',
            'department',
            'supplier',
            'fiscal_year',
            'total_price'
        ]
        
        for field in key_fields:
            mongo_value = str(mongo_record.get(field, ''))
            chroma_value = str(chroma_record.get(field, ''))
            
            if mongo_value != chroma_value:
                return False
                
        return True

def main():
    # Configuration
    config = {
        'mongo_uri': 'mongodb://localhost:27017/',
        'database': 'procurement_db',
        'collection': 'purchases',
        'persist_directory': './chroma_db'
    }
    
    # Initialize checker
    checker = RecordChecker(**config)
    
    try:
        # Check record counts
        print("\n=== Record Counts ===")
        stats = checker.check_records()
        print(f"MongoDB records: {stats['mongodb_count']}")
        print(f"ChromaDB records: {stats['chromadb_count']}")
        print(f"Difference: {stats['difference']}")
        
        # Verify record integrity
        print("\n=== Record Integrity Check ===")
        integrity = checker.verify_record_integrity(sample_size=1000)
        print(f"Records checked: {integrity['records_checked']}")
        print(f"Matching records: {integrity['matching_records']}")
        print(f"Missing records: {integrity['missing_records']}")
        print(f"Mismatched records: {integrity['mismatched_records']}")
        
        if integrity['issues']:
            print("\nIssues found:")
            for issue in integrity['issues'][:5]:  # Show first 5 issues
                print(f"- {issue['record_id']}: {issue['type']} - {issue['details']}")
        
    except Exception as e:
        print(f"Error checking records: {str(e)}")

if __name__ == "__main__":
    main()