from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Read extracted text
with open("data/pib_document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Better chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = text_splitter.split_text(text)

# Save chunks
os.makedirs("chunks", exist_ok=True)

for i, chunk in enumerate(chunks):
    with open(f"chunks/chunk_{i}.txt", "w", encoding="utf-8") as f:
        f.write(chunk)

print(f"Created {len(chunks)} chunks.")