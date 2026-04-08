#!/usr/bin/env python3
"""
Generate a styled PDF from PRESENTATION.md using fpdf2.
Parses markdown manually and renders with custom styling.
"""

import os
import re
from fpdf import FPDF

INPUT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRESENTATION.md")
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRESENTATION.pdf")

# ── Color palette ──
INDIGO = (79, 70, 229)
DARK = (15, 23, 42)
SLATE = (51, 65, 85)
LIGHT_SLATE = (100, 116, 139)
WHITE = (255, 255, 255)
LIGHT_BG = (248, 250, 252)
INDIGO_LIGHT = (238, 242, 255)
CODE_BG = (30, 41, 59)
CODE_FG = (226, 232, 240)
TABLE_HEADER_BG = (79, 70, 229)
TABLE_ALT_BG = (248, 250, 252)


FONT_DIR = '/System/Library/Fonts/Supplemental'


def sanitize(text):
    """Replace Unicode chars that may cause issues with safe ASCII equivalents."""
    replacements = {
        '\u2014': '--',  # em dash
        '\u2013': '-',   # en dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...',  # ellipsis
        '\u2192': '->',   # right arrow
        '\u2190': '<-',   # left arrow
        '\u2265': '>=',   # >=
        '\u2264': '<=',   # <=
        '\u00d7': 'x',    # multiplication
        '\u2022': '*',    # bullet
        '\u2713': '[v]',  # checkmark
        '\u2717': '[x]',  # X mark
        '\u221e': 'inf',  # infinity
        '\u03b1': 'alpha', # alpha
        '\u03b2': 'beta',  # beta
        '\u03b5': 'epsilon', # epsilon
        '\u03b3': 'gamma',   # gamma
        '\u03b4': 'delta',   # delta
        '\u03c9': 'omega',   # omega
        '\u03bb': 'lambda',  # lambda
        '\u03c0': 'pi',     # pi
        '\u03c3': 'sigma',  # sigma
        '\u03c4': 'tau',    # tau
        '\u03b7': 'eta',    # eta
        '\u03bc': 'mu',     # mu
        '\u2248': '~=',    # approximately
        '\u2260': '!=',    # not equal
        '\u221a': 'sqrt',  # square root
        '\u2211': 'SUM',   # summation
        '\u220f': 'PROD',  # product
        '\u2208': ' in ',  # element of
        '\u2282': 'subset', # subset
        '\u222b': 'integral', # integral
        '\u2202': 'd',       # partial
        '\u2207': 'nabla',   # nabla
        '\u00b2': '^2',     # superscript 2
        '\u00b3': '^3',     # superscript 3
        '\u2070': '^0',     # superscript 0
        '\u00b9': '^1',     # superscript 1
        '\u2074': '^4',
        '\u2075': '^5',
        '\u2076': '^6',
        '\u2077': '^7',
        '\u2078': '^8',
        '\u2079': '^9',
        '\u2080': '_0',
        '\u2081': '_1',
        '\u2082': '_2',
        '\u2083': '_3',
        '\u2084': '_4',
        '\u2085': '_5',
        '\u2086': '_6',
        '\u2087': '_7',
        '\u2088': '_8',
        '\u2089': '_9',
        '\u00b0': ' deg',  # degree
        '\u00b1': '+/-',   # plus-minus
        '\u2103': ' degC', # degree celsius
        '\u2109': ' degF', # degree fahrenheit
        '\u00ae': '(R)',   # registered
        '\u2122': '(TM)',  # trademark
        '\u00a9': '(C)',   # copyright
        '\u2020': '+',     # dagger
        '\u2021': '++',    # double dagger
        '\u00a7': 'S',     # section
        '\u21d2': '=>',    # double right arrow
        '\u21d0': '<=',    # double left arrow
        '\u2194': '<->',   # bidirectional arrow
        '\u2200': 'for all', # for all
        '\u2203': 'exists',  # exists
        '\u2205': 'empty',   # empty set
        '\u2261': '===',     # identical
        '\u221d': ' proportional to ', # proportional
        '\u2032': "'",      # prime
        '\u2033': "''",     # double prime
        '\u2500': '-',      # box-drawing horizontal
        '\u2502': '|',      # box-drawing vertical
        '\u250c': '+',      # box-drawing corner
        '\u2510': '+',
        '\u2514': '+',
        '\u2518': '+',
        '\u2534': '+',
        '\u252c': '+',
        '\u251c': '+',
        '\u2524': '+',
        '\u253c': '+',
        '\u2550': '=',
        '\u2551': '||',
        '\u2588': '#',      # full block
        '\u2591': '.',      # light shade
        '\u2592': ':',      # medium shade
        '\u2593': '#',      # dark shade
        '\u25b6': '>',      # right triangle
        '\u25c0': '<',      # left triangle
        '\u25cf': '*',      # black circle
        '\u25cb': 'o',      # white circle
        '\u2605': '*',      # star
        '\u2606': '*',      # white star
        '\u2610': '[ ]',    # ballot box
        '\u2611': '[x]',    # ballot box with check
        '\u2612': '[X]',    # ballot box with X
        '\u2660': 'S',      # spade
        '\u2663': 'C',      # club
        '\u2665': 'H',      # heart
        '\u2666': 'D',      # diamond
        '\u2580': '-',      # upper half block
        '\u2584': '_',      # lower half block
        '\u2588': '#',      # full block
        '\u2581': '_',      # lower 1/8 block
        '\u2582': '_',      # lower 1/4 block
        '\u2583': '_',      # lower 3/8 block  
        '\u2585': '-',      # lower 5/8 block
        '\u2586': '-',      # lower 3/4 block
        '\u2587': '-',      # lower 7/8 block
        '\u2501': '-',      # heavy horizontal
        '\u2503': '|',      # heavy vertical
        '\u250f': '+',
        '\u2513': '+',
        '\u2517': '+',
        '\u251b': '+',
        '\u2523': '+',
        '\u252b': '+',
        '\u2533': '+',
        '\u253b': '+',
        '\u254b': '+',
        '\u2578': '-',
        '\u2579': '|',
        '\u257a': '-',
        '\u257b': '|',
        '\u257c': '-',
        '\u257d': '|',
        '\u257e': '-',
        '\u257f': '|',
        '\u25b8': '>',      # small right triangle
        '\u25be': 'v',      # small down triangle
        '\u25b4': '^',      # small up triangle
        '\u25c2': '<',      # small left triangle
        '\u2503': '|',
        '\u2501': '-',
        '\u2595': '|',      # right 1/8 block
        '\u258f': '|',      # left 1/8 block
        '\u2594': '-',      # upper 1/8 block
        '\u2581': '_',      # lower 1/8 block  
        '\u23af': '-',      # horizontal line extension
        '\u23b8': '|',      # left vertical box line
        '\u23b9': '|',      # right vertical box line
        '\u23ce': '<enter>', # return symbol
        '\u23e9': '>>',     # fast forward
        '\u23ea': '<<',     # rewind
        '\u23ed': '|>',     # skip to end
        '\u23ee': '<|',     # skip to start
        '\u23ef': '>||',    # play/pause
        '\u23f8': '||',     # pause
        '\u23f9': '[]',     # stop
        '\u23fa': '(o)',    # record
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final pass: remove any remaining non-ASCII non-Latin chars
    result = []
    for ch in text:
        if ord(ch) < 128:
            result.append(ch)
        elif ord(ch) < 256:
            result.append(ch)  # Keep Latin-1 supplement
        else:
            result.append('?')  # Replace unknown
    return ''.join(result)


class PresentationPDF(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(*LIGHT_SLATE)
            self.cell(0, 5, sanitize('GFlowNet for Biodegradable Polymer Discovery'), align='L')
            self.ln(2)
            self.set_draw_color(*INDIGO)
            self.set_line_width(0.3)
            self.line(20, self.get_y(), 190, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(*LIGHT_SLATE)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def render_title(self, text):
        """Main document title - page 1."""
        self.ln(30)
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(*DARK)
        self.multi_cell(0, 12, sanitize(text), align='C')
        self.ln(3)
        # Accent line under title
        self.set_draw_color(*INDIGO)
        self.set_line_width(1)
        w = 60
        x = (210 - w) / 2
        self.line(x, self.get_y(), x + w, self.get_y())
        self.ln(6)
        # Subtitle
        self.set_font('Helvetica', 'I', 11)
        self.set_text_color(*INDIGO)
        self.cell(0, 8, 'Comprehensive Presentation Guide', align='C')
        self.ln(10)

    def render_h1(self, text):
        """Section heading (# heading)."""
        self.add_page()
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(*DARK)
        self.multi_cell(0, 10, sanitize(text))
        self.set_draw_color(*INDIGO)
        self.set_line_width(0.8)
        self.line(20, self.get_y() + 1, 190, self.get_y() + 1)
        self.ln(6)

    def render_h2(self, text):
        """Subsection heading (## heading)."""
        self.check_page_break(20)
        self.ln(4)
        # Left accent bar
        y = self.get_y()
        self.set_fill_color(*INDIGO)
        self.rect(20, y, 1.5, 8, 'F')
        self.set_x(25)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 64, 175)  # blue-800
        self.multi_cell(0, 8, sanitize(text))
        self.ln(3)

    def render_h3(self, text):
        """Sub-subsection heading (### heading)."""
        self.check_page_break(15)
        self.ln(3)
        self.set_font('Helvetica', 'B', 11.5)
        self.set_text_color(*SLATE)
        self.multi_cell(0, 7, sanitize(text))
        self.ln(2)

    def render_h4(self, text):
        """Sub-sub-subsection heading (#### heading)."""
        self.check_page_break(12)
        self.ln(2)
        self.set_font('Helvetica', 'B', 10.5)
        self.set_text_color(71, 85, 105)  # slate-600
        self.multi_cell(0, 6, sanitize(text))
        self.ln(1)

    def render_paragraph(self, text):
        """Regular paragraph text with inline formatting."""
        self.check_page_break(10)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*DARK)
        # Strip inline markdown
        clean = sanitize(self._strip_inline(text))
        self.multi_cell(0, 5.5, clean)
        self.ln(2)

    def render_bullet(self, text, level=0):
        """Bullet list item."""
        self.check_page_break(8)
        indent = 25 + level * 6
        self.set_x(indent)
        # Bullet marker
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*INDIGO)
        marker = '-' if level == 0 else '>'
        self.cell(5, 5.5, marker)
        # Content
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*DARK)
        clean = sanitize(self._strip_inline(text))
        remaining_w = 170 - (indent - 20) - 5
        self.multi_cell(remaining_w, 5.5, clean)
        self.ln(1)

    def render_numbered(self, num, text, level=0):
        """Numbered list item."""
        self.check_page_break(8)
        indent = 25 + level * 6
        self.set_x(indent)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*INDIGO)
        self.cell(7, 5.5, f'{num}.')
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*DARK)
        clean = sanitize(self._strip_inline(text))
        remaining_w = 170 - (indent - 20) - 7
        self.multi_cell(remaining_w, 5.5, clean)
        self.ln(1)

    def render_code_block(self, lines):
        """Render code block with dark background."""
        self.check_page_break(len(lines) * 4 + 10)
        self.ln(2)
        y_start = self.get_y()
        # Background
        text_content = '\n'.join(lines)
        # Estimate height
        self.set_font('Courier', '', 8)
        h = len(lines) * 4.2 + 8
        self.set_fill_color(*CODE_BG)
        # Left accent
        self.set_draw_color(*INDIGO)
        self.rect(20, y_start, 170, h, 'F')
        self.rect(20, y_start, 1.5, h, 'F')
        self.set_fill_color(*INDIGO)
        self.rect(20, y_start, 1.5, h, 'F')
        self.set_xy(24, y_start + 4)
        self.set_text_color(*CODE_FG)
        self.set_font('Courier', '', 8)
        for line in lines:
            if self.get_y() > 270:
                break
            self.set_x(24)
            self.cell(0, 4.2, sanitize(line[:120]))  # Truncate very long lines
            self.ln(4.2)
        self.set_y(y_start + h + 2)
        self.ln(2)

    def render_table(self, headers, rows):
        """Render table with styled header."""
        self.check_page_break(len(rows) * 7 + 15)
        self.ln(2)
        n_cols = len(headers)
        # Calculate column widths proportionally
        total_w = 170
        col_widths = self._calc_col_widths(headers, rows, total_w)

        # Header
        self.set_fill_color(*TABLE_HEADER_BG)
        self.set_text_color(*WHITE)
        self.set_font('Helvetica', 'B', 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, sanitize(h.strip()), border=0, fill=True, align='L')
        self.ln()

        # Rows
        self.set_font('Helvetica', '', 8.5)
        for r_idx, row in enumerate(rows):
            if self.get_y() > 265:
                self.add_page()
            if r_idx % 2 == 1:
                self.set_fill_color(*TABLE_ALT_BG)
            else:
                self.set_fill_color(*WHITE)
            self.set_text_color(*DARK)
            max_h = 6
            for i, cell_text in enumerate(row):
                w = col_widths[i] if i < len(col_widths) else 30
                clean = sanitize(self._strip_inline(cell_text.strip()))
                self.cell(w, max_h, clean[:60], border=0, fill=True, align='L')
            self.ln()
            # Bottom line
            self.set_draw_color(226, 232, 240)
            self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    def render_blockquote(self, text):
        """Render blockquote with accent bar."""
        self.check_page_break(12)
        self.ln(2)
        y = self.get_y()
        self.set_fill_color(*INDIGO_LIGHT)
        clean = sanitize(self._strip_inline(text))
        # Estimate height
        self.set_font('Helvetica', 'I', 9.5)
        h = max(12, (len(clean) // 80 + 1) * 5.5 + 6)
        self.rect(20, y, 170, h, 'F')
        self.set_fill_color(165, 180, 252)
        self.rect(20, y, 1.5, h, 'F')
        self.set_xy(25, y + 3)
        self.set_text_color(55, 48, 163)
        self.multi_cell(160, 5.5, clean)
        self.set_y(y + h + 2)
        self.ln(1)

    def render_hr(self):
        """Horizontal rule."""
        self.ln(4)
        y = self.get_y()
        self.set_draw_color(*INDIGO)
        self.set_line_width(0.4)
        self.line(60, y, 150, y)
        self.ln(4)

    def check_page_break(self, h):
        """Force page break if not enough room."""
        if self.get_y() + h > 275:
            self.add_page()

    def _strip_inline(self, text):
        """Remove markdown inline formatting for plain text rendering."""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # italic
        text = re.sub(r'`(.+?)`', r'\1', text)        # inline code
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # links
        return text

    def _calc_col_widths(self, headers, rows, total_w):
        """Calculate column widths based on content."""
        n = len(headers)
        if n == 0:
            return []
        # Estimate based on header + max cell length
        lengths = []
        for i in range(n):
            max_len = len(headers[i])
            for row in rows[:10]:
                if i < len(row):
                    max_len = max(max_len, len(row[i]))
            lengths.append(max_len)
        total_len = sum(lengths) or 1
        widths = [max(15, int(total_w * l / total_len)) for l in lengths]
        # Adjust to fit exactly
        diff = total_w - sum(widths)
        widths[-1] += diff
        return widths


def parse_and_render(pdf, md_text):
    """Parse markdown and render each element."""
    lines = md_text.split('\n')
    i = 0
    is_first_h1 = True

    while i < len(lines):
        line = lines[i]

        # ── Blank line ──
        if line.strip() == '':
            i += 1
            continue

        # ── Horizontal rule ──
        if line.strip() == '---' or line.strip() == '***':
            pdf.render_hr()
            i += 1
            continue

        # ── Headers ──
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            if is_first_h1:
                pdf.render_title(text)
                is_first_h1 = False
            else:
                pdf.render_h1(text)
            i += 1
            continue

        if line.startswith('## '):
            pdf.render_h2(line[3:].strip())
            i += 1
            continue

        if line.startswith('### '):
            pdf.render_h3(line[4:].strip())
            i += 1
            continue

        if line.startswith('#### '):
            pdf.render_h4(line[5:].strip())
            i += 1
            continue

        # ── Code block ──
        if line.strip().startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # Skip closing ```
            if code_lines:
                pdf.render_code_block(code_lines)
            continue

        # ── Table ──
        if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            # Parse table
            headers = [c.strip() for c in line.split('|') if c.strip()]
            i += 2  # Skip header and separator
            rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].split('|') if c.strip()]
                rows.append(cells)
                i += 1
            pdf.render_table(headers, rows)
            continue

        # ── Blockquote ──
        if line.startswith('> '):
            quote_text = line[2:].strip()
            i += 1
            while i < len(lines) and lines[i].startswith('> '):
                quote_text += ' ' + lines[i][2:].strip()
                i += 1
            pdf.render_blockquote(quote_text)
            continue

        # ── Bullet list ──
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            level = (len(line) - len(line.lstrip())) // 2
            text = re.sub(r'^[\s]*[-*]\s+', '', line)
            pdf.render_bullet(text, level)
            i += 1
            continue

        # ── Numbered list ──
        m = re.match(r'^(\s*)(\d+)\.\s+(.+)', line)
        if m:
            level = len(m.group(1)) // 2
            num = m.group(2)
            text = m.group(3)
            pdf.render_numbered(num, text, level)
            i += 1
            continue

        # ── Regular paragraph ──
        para = line.strip()
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith('#') \
                and not lines[i].startswith('```') and not lines[i].startswith('|') \
                and not lines[i].startswith('- ') and not lines[i].startswith('* ') \
                and not lines[i].startswith('> ') and not re.match(r'^\d+\.', lines[i]) \
                and lines[i].strip() != '---':
            para += ' ' + lines[i].strip()
            i += 1
        if para:
            pdf.render_paragraph(para)


def main():
    print("📖 Reading PRESENTATION.md...")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("📄 Generating styled PDF...")
    pdf = PresentationPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    parse_and_render(pdf, md_text)

    pdf.output(OUTPUT_PDF)
    size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
    print(f"✅ PDF generated: {OUTPUT_PDF} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
