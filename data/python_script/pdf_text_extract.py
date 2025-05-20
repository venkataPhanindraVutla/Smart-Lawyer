import fitz
import os

def extract_text_from_pdf(pdf_path, output_dir):
    """Extracts text from a PDF file and saves it as a text file."""
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save text to a file
        pdf_filename = os.path.basename(pdf_path)
        txt_filename = os.path.splitext(pdf_filename)[0] + ".txt"
        output_path = os.path.join(output_dir, txt_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Successfully extracted text from {pdf_path} to {output_path}")

    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")

if __name__ == "__main__":
    data_dir = "./data"
    output_dir = "./data/txt"
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    
    # Find all PDF files in the data directory
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        extract_text_from_pdf(pdf_file, output_dir)