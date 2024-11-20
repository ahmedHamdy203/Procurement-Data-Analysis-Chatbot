import pandas as pd
from pymongo import MongoClient
import logging
from datetime import datetime
import os
from typing import Optional, Dict, Any
import json
import numpy as np

class ProcurementDataLoader:
    def __init__(self, mongo_uri: str, database: str, collection: str):
        self.mongo_uri = mongo_uri
        self.database = database
        self.collection = collection
        self.client = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_loader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        try:
            self.client = MongoClient(self.mongo_uri)
            self.client.admin.command('ping')
            self.logger.info("Successfully connected to MongoDB")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False

    def clean_price(self, price_str: Any) -> float:
        """
        Clean price strings and convert to float.
        """
        if pd.isna(price_str):
            return 0.0
        if isinstance(price_str, (int, float)):
            return float(price_str)
        try:
            # Remove currency symbols and commas, then convert to float
            cleaned = str(price_str).replace('$', '').replace(',', '').strip()
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0

    def clean_quantity(self, qty: Any) -> float:
        """
        Clean quantity values and convert to float.
        """
        if pd.isna(qty):
            return 0.0
        try:
            return float(qty)
        except:
            return 0.0

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse dates with proper error handling and logging.
        """
        # Log sample values
        for col in ['Creation Date', 'Purchase Date']:
            if col in df.columns:
                self.logger.info(f"Sample {col} values: {df[col].head().tolist()}")

        # Parse Creation Date
        if 'Creation Date' in df.columns:
            df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
            null_dates = df['Creation Date'].isna().sum()
            self.logger.info(f"Null Creation Dates: {null_dates}")

        # Parse Purchase Date (optional)
        if 'Purchase Date' in df.columns:
            df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
            null_dates = df['Purchase Date'].isna().sum()
            self.logger.info(f"Null Purchase Dates: {null_dates}")

        return df

    def calculate_fiscal_year(self, date: pd.Timestamp) -> int:
        """
        Calculate fiscal year based on date.
        CA fiscal year starts July 1 and ends June 30.
        """
        if pd.isna(date):
            return 0
            
        year = date.year
        # If date is before July, it's in the previous fiscal year
        if date.month < 7:
            return year
        return year + 1

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data before loading into MongoDB.
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # Clean numeric fields first
        df['Total Price'] = df['Total Price'].apply(self.clean_price)
        df['Unit Price'] = df['Unit Price'].apply(self.clean_price)
        df['Quantity'] = df['Quantity'].apply(self.clean_quantity)
        
        # Handle categorical fields
        df['CalCard'] = df['CalCard'].fillna('NO').str.upper()
        df['CalCard'] = df['CalCard'].map({'YES': True, 'NO': False})
        
        # Fill missing values for string fields
        string_columns = [
            'Supplier Qualifications', 'Item Description', 'Item Name',
            'Classification Codes', 'Commodity Title', 'Class Title',
            'Family Title', 'Segment Title', 'Location', 'LPA Number',
            'Supplier Name', 'Department Name', 'Supplier Zip Code',
            'Sub-Acquisition Type', 'Sub-Acquisition Method'
        ]
        
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Handle numeric ID fields
        numeric_columns = ['Supplier Code', 'Normalized UNSPSC', 'Class', 'Family', 'Segment']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

        # Convert dates to ISO format strings
        if 'Creation Date' in df.columns:
            df['Creation Date'] = df['Creation Date'].apply(
                lambda x: x.isoformat() if pd.notnull(x) else None
            )
        
        if 'Purchase Date' in df.columns:
            df['Purchase Date'] = df['Purchase Date'].apply(
                lambda x: x.isoformat() if pd.notnull(x) else None
            )

        # Recalculate Fiscal Year based on Creation Date
        creation_dates = pd.to_datetime(df['Creation Date'], errors='coerce')
        df['Fiscal Year'] = creation_dates.apply(self.calculate_fiscal_year)

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the data before loading into MongoDB.
        """
        required_columns = [
            'Creation Date', 'Purchase Order Number', 
            'Department Name', 'Supplier Name',
            'Total Price'
        ]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False

        self.logger.info(f"Available columns: {df.columns.tolist()}")
            
        try:
            # Parse dates first
            df = self.parse_dates(df)
            
            # Check Creation Date (primary date field)
            creation_date_nulls = df['Creation Date'].isna().sum()
            if creation_date_nulls / len(df) > 0.1:  # Allow 10% missing
                self.logger.error(f"Too many invalid Creation Dates: {creation_date_nulls}")
                return False

            # Calculate Fiscal Year from Creation Date
            creation_dates = pd.to_datetime(df['Creation Date'], errors='coerce')
            df['Fiscal Year'] = creation_dates.apply(self.calculate_fiscal_year)
            fiscal_year_nulls = df['Fiscal Year'].isna().sum()
            self.logger.info(f"Null Fiscal Years after calculation: {fiscal_year_nulls}")

            # Check for null values in critical columns
            for col in ['Purchase Order Number', 'Department Name', 'Supplier Name']:
                null_count = df[col].isna().sum()
                self.logger.info(f"Null count for {col}: {null_count}")
                if null_count / len(df) > 0.1:  # Allow 10% missing values
                    self.logger.error(f"Too many null values in column {col}: {null_count} nulls")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False

    def load_data(self, file_path: str, batch_size: int = 1000) -> Dict[str, Any]:
        stats = {
            'total_records': 0,
            'successful_inserts': 0,
            'failed_inserts': 0,
            'start_time': datetime.now()
        }

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            self.logger.info(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
            stats['total_records'] = len(df)

            # Log sample data
            self.logger.info(f"First few rows of the dataframe:\n{df.head().to_string()}")
            self.logger.info(f"Data types of columns:\n{df.dtypes}")

            if not self.validate_data(df):
                raise ValueError("Data validation failed")

            df = self.preprocess_data(df)

            if not self.connect():
                raise ConnectionError("Failed to connect to MongoDB")

            db = self.client[self.database]
            collection = db[self.collection]

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                try:
                    records = batch.to_dict('records')
                    collection.insert_many(records)
                    stats['successful_inserts'] += len(records)
                    self.logger.info(f"Inserted batch of {len(records)} records")
                except Exception as e:
                    stats['failed_inserts'] += len(batch)
                    self.logger.error(f"Failed to insert batch: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
        finally:
            if self.client:
                self.client.close()
            stats['end_time'] = datetime.now()
            stats['duration'] = stats['end_time'] - stats['start_time']
            
            self.logger.info(f"""
            Data Loading Summary:
            - Total records: {stats['total_records']}
            - Successful inserts: {stats['successful_inserts']}
            - Failed inserts: {stats['failed_inserts']}
            - Duration: {stats['duration']}
            """)
            
        return stats

def main():
    config = {
        'mongo_uri': 'mongodb://localhost:27017/',
        'database': 'procurement_db',
        'collection': 'purchases',
        'csv_file': './data/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv'
    }
    
    loader = ProcurementDataLoader(
        mongo_uri=config['mongo_uri'],
        database=config['database'],
        collection=config['collection']
    )
    
    try:
        stats = loader.load_data(config['csv_file'])
        print(json.dumps(stats, default=str, indent=2))
    except Exception as e:
        print(f"Failed to load data: {str(e)}")

if __name__ == "__main__":
    main()