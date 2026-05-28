import fitz

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        pages.append({
            "page": page_num + 1,
            "text": text
        })

    return pages