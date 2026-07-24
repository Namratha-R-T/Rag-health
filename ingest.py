import fitz  # PyMuPDF

pdf_path = "data/pib_document.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

with open("data/pib_document.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("PDF text extracted successfully.")