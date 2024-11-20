import pandas as pd
from pymongo import MongoClient
from pprint import pprint
from typing import Optional, List, Dict, Any

class MongoDBViewer:
    def __init__(self, mongo_uri: str, database: str, collection: str):
        """
        Initialize MongoDB viewer
        
        Args:
            mongo_uri (str): MongoDB connection string
            database (str): Database name
            collection (str): Collection name
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database]
        self.collection = self.db[collection]

    def get_sample_documents(self, n: int = 5) -> List[Dict]:
        """
        Get a sample of documents from the collection
        
        Args:
            n (int): Number of documents to retrieve
            
        Returns:
            List of documents
        """
        return list(self.collection.find().limit(n))

    def get_collection_stats(self) -> Dict:
        """
        Get basic statistics about the collection
        
        Returns:
            Dictionary containing collection statistics
        """
        return {
            'total_documents': self.collection.count_documents({}),
            'distinct_departments': len(self.collection.distinct('Department Name')),
            'distinct_suppliers': len(self.collection.distinct('Supplier Name')),
            'fiscal_years': self.collection.distinct('Fiscal Year')
        }

    def search_by_department(self, department_name: str, limit: int = 5) -> List[Dict]:
        """
        Search documents by department name
        
        Args:
            department_name (str): Department name to search for
            limit (int): Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        return list(self.collection.find(
            {'Department Name': {'$regex': department_name, '$options': 'i'}}
        ).limit(limit))

    def get_total_spend_by_department(self) -> List[Dict]:
        """
        Get total spending aggregated by department
        
        Returns:
            List of departments and their total spending
        """
        pipeline = [
            {
                '$group': {
                    '_id': '$Department Name',
                    'total_spend': {'$sum': '$Total Price'},
                    'number_of_orders': {'$sum': 1}
                }
            },
            {
                '$sort': {'total_spend': -1}
            }
        ]
        return list(self.collection.aggregate(pipeline))

def main():
    # Configuration
    config = {
        'mongo_uri': 'mongodb://localhost:27017/',
        'database': 'procurement_db',
        'collection': 'purchases'
    }
    
    # Initialize viewer
    viewer = MongoDBViewer(**config)
    
    # Get collection statistics
    print("\n=== Collection Statistics ===")
    stats = viewer.get_collection_stats()
    pprint(stats)
    
    # Show sample documents
    print("\n=== Sample Documents ===")
    samples = viewer.get_sample_documents(2)
    pprint(samples)
    
    # Show department spending
    print("\n=== Department Spending ===")
    dept_spending = viewer.get_total_spend_by_department()
    pprint(list(dept_spending)[:5])  # Show top 5 departments

if __name__ == "__main__":
    main()