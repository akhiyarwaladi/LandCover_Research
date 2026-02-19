"""Convert the manuscript markdown to a journal-style PDF using markdown_pdf.

Usage: python convert_to_pdf.py
Output: MANUSCRIPT_Land_Cover_Classification_Jambi.pdf in same directory.
"""
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(SCRIPT_DIR, "MANUSCRIPT_Land_Cover_Classification_Jambi.md")
PDF_FILE = os.path.join(SCRIPT_DIR, "MANUSCRIPT_Land_Cover_Classification_Jambi.pdf")
FIG_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "results", "figures"))
# Temporary directory for resized images
PDF_FIGS = os.path.join(SCRIPT_DIR, "_pdf_figures")
MAX_IMG_WIDTH = 1800  # pixels, sufficient for print quality

# Journal-style CSS
CSS = """
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.45;
    text-align: justify;
}
h1 {
    font-size: 15pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 6pt;
}
h2 {
    font-size: 12pt;
    font-weight: bold;
    text-align: center;
    margin-top: 16pt;
    margin-bottom: 8pt;
}
h3 {
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 12pt;
    margin-bottom: 4pt;
}
h4 {
    font-size: 11pt;
    font-style: italic;
    margin-top: 8pt;
    margin-bottom: 4pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 9pt;
    margin: 8pt 0;
}
th {
    font-weight: bold;
    padding: 3pt 5pt;
    border-top: 2px solid #000;
    border-bottom: 1px solid #000;
    text-align: center;
}
td {
    padding: 2pt 5pt;
    text-align: center;
}
tbody tr:last-child td {
    border-bottom: 2px solid #000;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10pt auto;
}
hr {
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 14pt 0;
}
"""


def find_figure_file(filename, search_dirs):
    """Search for a figure file in multiple directories."""
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def resize_image(src, dest, max_width):
    """Resize image to max_width while preserving aspect ratio."""
    from PIL import Image
    img = Image.open(src)
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"  Resized: {os.path.basename(src)} ({img.width}x{img.height})")
    img.save(dest, optimize=True, quality=85)


def convert():
    from markdown_pdf import MarkdownPdf, Section

    os.makedirs(PDF_FIGS, exist_ok=True)

    print(f"Reading {MD_FILE}...")
    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Collect all subdirectories under FIG_SRC for image search
    fig_dirs = [FIG_SRC]
    for root, dirs, files in os.walk(FIG_SRC):
        for d in dirs:
            fig_dirs.append(os.path.join(root, d))

    # Copy and resize referenced images to temp dir for PDF embedding
    def simplify_img_path(match):
        alt = match.group(1)
        rel_path = match.group(2)
        filename = os.path.basename(rel_path)

        source = find_figure_file(filename, fig_dirs)
        if source:
            dest = os.path.join(PDF_FIGS, filename)
            if not os.path.exists(dest):
                resize_image(source, dest, MAX_IMG_WIDTH)
            else:
                print(f"  Found:  {filename}")
        else:
            print(f"  MISSING: {filename}")

        return f"![{alt}]({filename})"

    md_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", simplify_img_path, md_text)

    print("\nCreating PDF...")
    pdf = MarkdownPdf(toc_level=0)
    pdf.meta["title"] = "Land Cover Classification Jambi Province"
    pdf.meta["author"] = ""

    section = Section(
        md_text,
        toc=False,
        root=PDF_FIGS,
        paper_size="A4",
        borders=(36, 36, -36, -36),
    )
    pdf.add_section(section, user_css=CSS)
    pdf.save(PDF_FILE)

    file_size = os.path.getsize(PDF_FILE)
    print(f"\nDone! PDF size: {file_size / 1024:.0f} KB ({file_size / (1024*1024):.1f} MB)")
    print(f"Output: {PDF_FILE}")


if __name__ == "__main__":
    convert()
