def recursive_chunking(text, size=200, separators=("\n\n", "\n", ". ", " ")):
    """Split on paragraphs first, then lines, sentences and words, only when a piece is too big."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    if not separators:                       # nothing left to split on, cut by characters
        return [text[i:i + size] for i in range(0, len(text), size)]

    sep, smaller_seps = separators[0], separators[1:]
    chunks = []
    current = ""
    for piece in text.split(sep):
        if len(piece) > size:                # piece too big, split it with the next separator
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(recursive_chunking(piece, size, smaller_seps))
        elif len(current) + len(sep) + len(piece) <= size:
            current = current + sep + piece if current else piece   # still fits, keep adding
        else:
            chunks.append(current.strip())   # full, save it and start a new one
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks