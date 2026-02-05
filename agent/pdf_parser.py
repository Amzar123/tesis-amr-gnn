import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

class FinancialDocAgent:
    def __init__(self, file_path):
        self.file_path = file_path

    def process_document(self):
        """
        Convert text and table to string format
        """
        all_content = []
        with pdfplumber.open(self.file_path) as pdf:
            doc = fitz.open(self.file_path)
            
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()

                table_strings = ""
                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    table_strings += df.to_string(index=False) + "\n"

                fitz_page = doc.load_page(i)
                blocks = fitz_page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))
                
                page_text = ""
                for b in blocks:
                    page_text += b[4].strip() + " "
                
                all_content.append({
                    "page": i + 1,
                    "narration": page_text,
                    "tables": table_strings
                })
        
        return all_content