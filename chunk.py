import os
import re

MIN_WORDS = 200
MAX_WORDS = 500

# Known section/topic keywords from the PIB "India's Health Transformation" page.
# Used to detect natural section boundaries so chunks stay topically coherent
# instead of being cut at arbitrary character counts.
SECTION_KEYWORDS = [
    "Ayushman Bharat", "AB-PMJAY", "PM-JAY",
    "Ayushman Arogya Mandir",
    "PM-ABHIM",
    "ABDM", "Ayushman Bharat Digital Mission",
    "NHM", "National Health Mission",
    "Health and Wellness Centre",
    "PMBJP", "Jan Aushadhi",
    "Poshan Abhiyaan",
    "Mission Indradhanush",
]


def looks_like_heading(line):
    """A short, title-like line (or one containing a known section keyword)
    is treated as the start of a new section."""
    line = line.strip()
    if not line:
        return False
    if any(kw.lower() in line.lower() for kw in SECTION_KEYWORDS):
        return True
    word_count = len(line.split())
    return word_count <= 12 and line[0].isupper() and not line.endswith(".")


def split_into_sections(text):
    """Split raw text into (title, body) sections using blank-line separated
    paragraphs, with the first line of a paragraph treated as its heading if
    it looks like one."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    sections = []
    current_title = "Introduction"
    current_body = []

    for para in paragraphs:
        lines = para.split("\n")
        first_line = lines[0].strip()

        if looks_like_heading(first_line) and len(lines) > 1:
            # Flush current section, start a new one titled after this heading
            if current_body:
                sections.append((current_title, "\n\n".join(current_body)))
            current_title = first_line
            current_body = ["\n".join(lines[1:]).strip()]
        else:
            current_body.append(para)

    if current_body:
        sections.append((current_title, "\n\n".join(current_body)))

    return sections


def enforce_word_bounds(sections):
    """Merge short sections together and split oversized ones so every
    resulting chunk falls within MIN_WORDS-MAX_WORDS, while keeping each
    chunk's title for reference."""
    chunks = []
    buffer_title = None
    buffer_text = ""

    def word_count(s):
        return len(s.split())

    def flush():
        nonlocal buffer_title, buffer_text
        if buffer_text.strip():
            chunks.append((buffer_title or "Introduction", buffer_text.strip()))
        buffer_title = None
        buffer_text = ""

    for title, body in sections:
        candidate = (buffer_text + "\n\n" + body).strip() if buffer_text else body

        if word_count(candidate) <= MAX_WORDS:
            # Fits in the current buffer (or starts a new one)
            buffer_text = candidate
            if buffer_title is None:
                buffer_title = title
            continue

        # Adding this section would overflow the buffer.
        if buffer_text:
            flush()

        if word_count(body) <= MAX_WORDS:
            buffer_title = title
            buffer_text = body
        else:
            # Section itself is too long on its own -> split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", body)
            piece = ""
            for sent in sentences:
                trial = (piece + " " + sent).strip() if piece else sent
                if word_count(trial) > MAX_WORDS and piece:
                    chunks.append((title, piece.strip()))
                    piece = sent
                else:
                    piece = trial
            if piece.strip():
                buffer_title = title
                buffer_text = piece.strip()

    flush()

    # Merge any final chunk that's under MIN_WORDS into the previous one,
    # if that keeps the previous chunk under MAX_WORDS.
    merged = []
    for title, body in chunks:
        if merged and word_count(body) < MIN_WORDS:
            prev_title, prev_body = merged[-1]
            combined = prev_body + "\n\n" + body
            if word_count(combined) <= MAX_WORDS:
                merged[-1] = (prev_title, combined)
                continue
        merged.append((title, body))

    return merged


def main():
    with open("data/pib_document.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sections = split_into_sections(text)
    chunks = enforce_word_bounds(sections)

    os.makedirs("chunks", exist_ok=True)

    for i, (title, body) in enumerate(chunks):
        with open(f"chunks/chunk_{i}.txt", "w", encoding="utf-8") as f:
            # Store the section title as the first line so it can be shown
            # in the UI as "source section" without a separate metadata file.
            f.write(f"[{title}]\n\n{body}")

    word_counts = [len(body.split()) for _, body in chunks]
    print(f"Created {len(chunks)} chunks.")
    print(f"Word counts -> min: {min(word_counts)}, max: {max(word_counts)}, "
          f"avg: {sum(word_counts) // len(word_counts)}")


if __name__ == "__main__":
    main()