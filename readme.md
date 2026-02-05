# AMR-GNN: Thesis Project on Abstract Meaning Representation with Graph Neural Networks

## Project Description

This thesis project explores the integration of **Abstract Meaning Representation (AMR)** with **Graph Neural Networks (GNN)** for processing and analyzing financial documents. The system extracts structured information from PDF documents using AMR parsing and leverages GNNs to compute importance scores for semantic entities.

### Key Objectives
- Extract text and tabular data from financial documents (PDFs)
- Convert AMR graphs to PyTorch Geometric data structures
- Apply Graph Neural Networks for semantic analysis
- Integrate with Retrieval-Augmented Generation (RAG) and language models via LangChain

---

## Project Structure

```
tesis-amr-gnn/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── readme.md                   # This file
├── AADI_2024_Annual_Report.pdf # Sample financial document
├── agent/
│   └── pdf_parser.py          # PDF extraction and parsing agent
└── venv/                       # Virtual environment
```

### Directory Breakdown

- **`app.py`**: Main application script that orchestrates the document processing pipeline and demonstrates the workflow
- **`agent/`**: Contains specialized modules for processing different document types
  - **`pdf_parser.py`**: `FinancialDocAgent` class for extracting text and tables from PDF files
- **`requirements.txt`**: Complete list of Python package dependencies

---

## Technology Stack

### Core Libraries
- **amrlib** (v0.8.0) - AMR parsing and graph conversion
- **PyTorch Geometric** - Graph Neural Network implementations
- **LangChain** (v0.4.1+) - LLM integration and RAG framework
- **FAISS** (v1.13.2) - Vector similarity search

### Document Processing
- **pdfplumber** - Extract tables and structured data from PDFs
- **PyMuPDF (fitz)** - Advanced PDF text extraction
- **pandas** - Data manipulation and formatting

### Supporting Libraries
- **langsmith** - LangChain monitoring and evaluation
- **huggingface_hub** - Pre-trained model access
- **cryptography** - Secure data handling

---

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tesis-amr-gnn
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Basic Workflow

The main entry point demonstrates the complete pipeline:

```bash
python app.py
```

### Step-by-Step Process

1. **Initialize the Financial Document Agent**
   ```python
   import agent.pdf_parser as pdf_parser
   
   agent = pdf_parser.FinancialDocAgent("AADI_2024_Annual_Report.pdf")
   ```

2. **Extract Document Content**
   ```python
   data = agent.process_document()
   ```

3. **Access Extracted Data**
   Each page returns a dictionary with:
   - `page`: Page number
   - `narration`: Extracted text content
   - `tables`: Formatted table data
   
   ```python
   for item in data:
       print(f"Page {item['page']}:")
       print(f"Text: {item['narration']}")
       print(f"Tables:\n{item['tables']}")
   ```

### Planned Features (Under Development)

The codebase includes commented-out implementations for:
- **GNN Model**: Graph Convolutional Networks for importance scoring
- **AMR to PyG Converter**: Converting AMR graphs to PyTorch Geometric Data objects
- **RAG Integration**: Combining extracted content with language models

---

## Key Components

### FinancialDocAgent (`agent/pdf_parser.py`)

Extracts text and tables from PDF documents.

**Methods:**
- `__init__(file_path)`: Initialize with PDF file path
- `process_document()`: Extract all page content (text + tables)

**Returns:**
```python
[
    {
        "page": 1,
        "narration": "Extracted text content...",
        "tables": "Formatted table data..."
    },
    # ... more pages
]
```

---

## Dependencies Overview

### Key Packages
- **amrlib**: Abstract Meaning Representation parsing
- **langchain-community/core**: LLM framework and utilities
- **torch-geometric**: Graph neural network operations
- **faiss-cpu**: Vector search and similarity
- **pdfplumber & fitz**: PDF processing

For complete dependencies, see [requirements.txt](requirements.txt)

---

## Development

### Running the Application
```bash
python app.py
```

### Expected Output
The script processes the sample financial report and outputs extracted data for each page with text and table content.

---

## Notes

- The project uses a virtual environment (`venv/`) for dependency isolation
- Sample document: `AADI_2024_Annual_Report.pdf` is included for testing
- GNN and AMR conversion components are under active development
- RAG integration with LangChain is planned for future iterations

---

## Future Improvements

- [ ] Implement full AMR graph parsing and conversion
- [ ] Complete GNN model training pipeline
- [ ] Integrate with language models via LangChain
- [ ] Add support for multiple document formats
- [ ] Implement evaluation metrics for importance scoring
- [ ] Add unit tests and documentation

---

## Author

Thesis project for ITB (Institut Teknologi Bandung)

---

## License

[Specify your license here]