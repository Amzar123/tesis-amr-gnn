import amrlib

# 1. Download/Load the model 
# Note: On first run, you may need to do    wnload the model manually 
# amrlib.download_stog_model()

def convert_text_to_amr(text):
    # Load the Segmenter (AMR works best on a sentence-by-sentence basis)
    # This uses spacy to break your PDF text into clean sentences
    device = 'cpu' # Change to 'cuda' if you have a GPU
    stog = amrlib.load_stog_model(device=device)
    
    # Split text into sentences
    # AMR models are usually trained on individual sentences
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
    
    # Generate AMR Graphs
    graphs = stog.parse_sents(sentences)
    
    return zip(sentences, graphs)

# Example Usage
pdf_text = "The Department of Justice filed a lawsuit against the corporation."

processed_data = convert_text_to_amr(pdf_text)

for sentence, amr_graph in processed_data:
    print(f"Sentence: {sentence}")
    print(f"AMR Graph:\n{amr_graph}\n")
    print("-" * 30)