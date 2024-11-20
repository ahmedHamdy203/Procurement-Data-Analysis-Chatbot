
## Table of Contents
- [Introduction](#introduction)
  - [Modules](#modules)
    - [1. `conversation_handler.py`](#1-conversation_handlerpy)
    - [2. `data_explorer.py`](#2-data_explorerpy)
    - [3. `data_integrity_checker.py`](#3-data_integrity_checkerpy)
    - [4. `data_loader_csv_to_mongo.py`](#4-data_loader_csv_to_mongopy)
    - [5. `vectorDB_handler.py`](#5-vectordb_handlerpy)
  - [Usage](#usage)
  - [How to start?](#how-to-start)

# Introduction

This project provides a set of Python modules for analyzing procurement data using MongoDB, ChromaDB, and various NLP and data processing libraries. The system enables loading data from CSV files into MongoDB, exploring and validating the data, and setting up a retrieval-augmented generation (RAG) pipeline for question-answering based on the procurement data.

## Modules

### 1. `conversation_handler.py`

This module defines the `RAGPipeline` class, which sets up and manages the RAG pipeline for question-answering. It includes the following key components:
- `ModelManager`: Downloads and manages the language model used for question-answering.
- `setup_embeddings`: Initializes the embedding model for vector representation of text.
- `setup_vectorstore`: Initializes the ChromaDB vector store for storing and retrieving documents.
- `setup_llm`: Initializes the language model for generating answers.
- `setup_qa_chain`: Sets up the question-answering chain with a custom prompt template.
- `query`: Processes a query by retrieving relevant documents and generating an answer.

### 2. `data_explorer.py`

This module provides the `MongoDBViewer` class for exploring and analyzing the procurement data stored in MongoDB. It includes methods for:
- Retrieving sample documents from the collection
- Getting basic statistics about the collection
- Searching documents by department name
- Aggregating total spending by department

### 3. `data_integrity_checker.py`

The `RecordChecker` class in this module is responsible for checking the integrity and consistency of records between MongoDB and ChromaDB. It provides methods for:
- Comparing record counts between the two databases
- Verifying the integrity of records by checking key fields
- Logging and reporting any discrepancies or issues found during the verification process

### 4. `data_loader_csv_to_mongo.py`

This module defines the `ProcurementDataLoader` class, which handles loading procurement data from a CSV file into MongoDB. It includes methods for:
- Connecting to MongoDB
- Preprocessing and cleaning the data
- Validating the data before loading
- Loading the data into MongoDB in batches
- Logging and reporting the progress and status of the data loading process

### 5. `vectorDB_handler.py`

The `ProcurementRAG` class in this module sets up and manages the retrieval-augmented generation (RAG) pipeline specifically for procurement data. It includes methods for:
- Formatting and creating metadata for procurement records
- Loading data from MongoDB into ChromaDB vector store
- Performing similarity search on the vector store with optional filters
- Retrieving filtered search results based on department, fiscal year, and price range

## Usage

1. Set up the required dependencies and configurations for each module.
2. Use `data_loader_csv_to_mongo.py` to load procurement data from a CSV file into MongoDB.
3. Explore and analyze the loaded data using `data_explorer.py`.
4. Verify the integrity and consistency of records between MongoDB and ChromaDB using `data_integrity_checker.py`.
5. Set up the RAG pipeline for question-answering using `conversation_handler.py` and `vectorDB_handler.py`.
6. Perform similarity searches and retrieve relevant documents based on queries using the methods provided in `vectorDB_handler.py`.

## How to start?
1. Create Mongo DB container using the following command:  
   ``
   docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v D:\case-study-penny\mongodb:/data/db:/data/db \
  mongo:latest
   ``
2. Create Environment using the following command:  
   `
    - conda env create -f environment.yml
    - conda activate proc_case_study
    - python ./conversation_handler.py
   `
