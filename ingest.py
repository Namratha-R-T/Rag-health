import os
import sys
import fitz  # PyMuPDF


def main():
    pdf_path = "data/pib_document.pdf"
    output_path = "data/pib_document.txt"

    if not os.path.exists(pdf_path):
        print(f"Error: Required input file '{pdf_path}' not found.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    doc = fitz.open(pdf_path)

    text = ""
    for page in doc:
        text += page.get_text()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("PDF text extracted successfully.")


if __name__ == "__main__":
    main()