"""Cutting narration text into subtitle lines (shared by the narration builders)."""

import re

SUBTITLE_LINE_MAX_CHARACTERS = 64


def subtitle_lines(text):
    """The narration cut into lines short enough to read: at sentence ends first, then at
    commas and colons, then between words."""
    lines = []
    for sentence in re.split(r"(?<=[.?!])\s+", text.strip()):
        pieces = [sentence]
        for pattern in (r"(?<=[,:;])\s+", r"\s+"):
            next_pieces = []
            for piece in pieces:
                if len(piece) <= SUBTITLE_LINE_MAX_CHARACTERS:
                    next_pieces.append(piece)
                    continue
                current = ""
                for part in re.split(pattern, piece):
                    candidate = f"{current} {part}".strip()
                    if current and len(candidate) > SUBTITLE_LINE_MAX_CHARACTERS:
                        next_pieces.append(current)
                        current = part
                    else:
                        current = candidate
                if current:
                    next_pieces.append(current)
            pieces = next_pieces
        lines.extend(pieces)
    return lines



CHINESE_LINE_MAX_CHARACTERS = 24


def subtitle_lines_chinese(text):
    """Chinese narration cut into subtitle lines: at the end of a sentence first, then at
    commas and colons, never inside a clause unless the clause alone is too long."""
    lines = []
    for sentence in re.findall(r"[^。？！]+[。？！]?", text.strip()):
        clauses = re.findall(r"[^，：；、]+[，：；、]?", sentence)
        current = ""
        for clause in clauses:
            if current and len(current) + len(clause) > CHINESE_LINE_MAX_CHARACTERS:
                lines.append(current)
                current = clause
            else:
                current += clause
            while len(current) > CHINESE_LINE_MAX_CHARACTERS + 6:
                lines.append(current[:CHINESE_LINE_MAX_CHARACTERS])
                current = current[CHINESE_LINE_MAX_CHARACTERS:]
        if current:
            lines.append(current)
    return [line.strip() for line in lines if line.strip()]
