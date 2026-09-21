"""Compile supplements/generated/*.md into submission-ready PDFs.

No pandoc in this environment, so this converts the constrained Markdown grammar
that `publication`/the supplement generator emits - headings, paragraphs,
inline code, and pipe tables - into LaTeX, then runs pdflatex.

The grammar is deliberately narrow because we generate the input ourselves.
Anything outside it raises rather than silently producing a wrong PDF.

Superseded PDFs in supplements/superseded_pdfs/ are never read.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
           "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
FIGURES = {
    "A": [("results/f1_highlights/figures/figA_highlight_thematic_structure.pdf",
           "Thematic structure of Formula 1 highlight commentary. Descriptive "
           "two-dimensional projection; clustering was performed in the "
           "five-component reduced space, not in this plane.")],
    "B": [("results/f1_highlights/figures/figB_category_diagnostic_profile.pdf",
           "Category-level diagnostic profile across four complementary "
           "dimensions, shown as computed. No aggregate vote or verdict band.")],
    "D": [("results/f1_full/figures/figC_fullrace_granularity_landscape.pdf",
           "Full-race granularity landscape; the selected setting is marked."),
          ("results/f1_cross_register/figures/figD_cross_register_correspondence.pdf",
           "Highlight x full-race centroid cosine with correspondence type."),
          ("results/f1_cross_register/figures/figE_matched_event_availability.pdf",
           "Theme availability in the ten matched events.")],
    "E": [("results/f1_cross_register/figures/figF_paired_category_effects.pdf",
           "Per-category paired effects; each dot is one of ten events."),
          ("results/f1_cross_register/figures/figG_matched_length_null.pdf",
           "Observed event-level contrast against the matched-length null."),
          ("results/f1_cross_register/figures/figG2_compositional_race_dynamics.pdf",
           "Race dynamics does not survive once Strategy is partitioned out.")],
}
PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage{booktabs,longtable,array,graphicx,microtype,tabularx,caption}
\usepackage{ragged2e}
\uchyph=1
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{hyperref}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\footnotesize What Commentators Talk About --- Supplementary Materials}
\fancyhead[R]{\footnotesize Supplement %(part)s}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
%% Long snake_case identifiers are unbreakable in \texttt, which is the sole
%% cause of the overfull boxes. \idt inserts a discretionary break after every
%% underscore, so identifiers wrap instead of running into the margin. No
%% character of any identifier is changed.
\newcommand{\idt}[1]{\texttt{\seqallowbreak #1}}
\newcommand{\seqallowbreak}{}
\makeatletter
%% capture the ORIGINAL \_ before re-letting it, else \ubreak recurses
\let\origunderscore\_
\def\ubreak{\origunderscore\allowbreak}
\makeatother
\sloppy
\setlength{\emergencystretch}{3em}
\raggedbottom
\title{\vspace{-14mm}\large Supplement %(part)s --- %(title)s}
\author{}\date{}
\begin{document}\maketitle\thispagestyle{fancy}
"""


def esc(s: str) -> str:
    out = []
    for ch in str(s):
        out.append(SPECIAL.get(ch, ch))
    return "".join(out)


def inline(s: str) -> str:
    """Handle `code` and **bold** before escaping the rest."""
    parts, last = [], 0
    for m in re.finditer(r"`([^`]+)`|\*\*([^*]+)\*\*", s):
        parts.append(esc(s[last:m.start()]))
        if m.group(1) is not None:
            parts.append(r"\texttt{" + esc(m.group(1)).replace(r"\_", r"\_\allowbreak{}") + "}")
        else:
            parts.append(r"\textbf{" + esc(m.group(2)) + "}")
        last = m.end()
    parts.append(esc(s[last:]))
    return "".join(parts)


LONG_TOKEN = re.compile(r"(?!x)x")   # disabled: see _breakable


def _breakable(tex: str) -> str:
    """Insert \\allowbreak inside long alphabetic tokens so narrow table cells
    can wrap them. Alphabetic-only, so numeric values are never split, and
    \\allowbreak prints nothing, so the value is unchanged."""
    # Mid-word splitting is deliberately NOT used: it produced breaks such as
    # WELL_REPRESENTE / D. Column widths are allocated proportionally instead,
    # and tokens still break at underscores via \\ubreak.
    return tex


def md_table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]          # cells[1] is the --- separator
    n = len(header)
    body = [r + [""] * (n - len(r)) if len(r) < n else r[:n] for r in body]
    size = r"\tiny" if n > 6 else r"\scriptsize"
    # longest unbreakable run per column (tokens break at "_", so split there)
    def _longest(col_cells):
        best = 1
        for c in col_cells:
            for tok in str(c).replace("_", " ").split():
                best = max(best, len(tok))
        return best
    widths = [_longest([header[j]] + [r[j] for r in body]) for j in range(n)]
    total = sum(widths)
    # clamp so no column is starved and none hogs the line
    share = [max(0.06, min(0.34, w / total)) for w in widths]
    share = [x / sum(share) for x in share]
    col = "".join(
        ">{\\RaggedRight\\arraybackslash\\let\\_\\ubreak}"
        "p{\\dimexpr%.4f\\textwidth-2\\tabcolsep\\relax}" % (0.97 * f)
        for f in share)
    out = [size, r"\begin{longtable}{" + col + "}", r"\toprule"]
    out.append(" & ".join(r"\textbf{" + _breakable(inline(h)) + "}"
                          for h in header) + r" \\")
    out.append(r"\midrule\endhead")
    for r in body:
        out.append(" & ".join(_breakable(inline(c)) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{longtable}", r"\normalsize", ""]
    return "\n".join(out)


def md_to_tex(md: str, part: str, title: str, rel_root: Path) -> str:
    lines = md.splitlines()
    body, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            i += 1
            continue                              # title comes from \maketitle
        if ln.startswith("## "):
            body.append(r"\subsection*{" + inline(ln[3:].strip()) + "}")
            i += 1
            continue
        if ln.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) >= 3:
                body.append(md_table(block))
            continue
        if ln.strip():
            txt = inline(ln.strip())
            if ln.strip().startswith("Columns:"):
                # a long comma-separated identifier list: set ragged-right so it
                # wraps cleanly instead of producing overfull boxes
                body.append("{" + "\\raggedright " + txt + "\\par}")
            else:
                body.append(txt)
            body.append("")
        i += 1

    figs = []
    for rel, cap in FIGURES.get(part, []):
        src = rel_root / rel
        pdf = src if src.exists() else src.with_suffix(".png")
        if pdf.exists():
            figs.append(r"\begin{figure}[htbp]\centering" "\n"
                        r"\includegraphics[width=0.92\textwidth,"
                        r"height=0.80\textheight,keepaspectratio]{%s}" %
                        pdf.as_posix() + "\n"
                        r"\caption*{\footnotesize " + esc(cap) + "}\n"
                        r"\end{figure}" "\n" r"\clearpage")
    if figs:
        body.append(r"\subsection*{Figures}")
        body += figs
    return (PREAMBLE % {"part": part, "title": esc(title)}
            + "\n".join(body) + "\n" + r"\end{document}" + "\n")


def build(rel_root: Path) -> list[dict]:
    rel_root = Path(rel_root)
    gen = rel_root / "supplements/generated"
    work = rel_root / "supplements/generated/_tex"
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for src in sorted(gen.glob("supplement_*.md")):
        part = src.stem.split("_")[1]
        title = src.stem.split("_", 2)[2].replace("_", " ").title()
        tex = md_to_tex(src.read_text(encoding="utf-8"), part, title, rel_root)
        texf = work / f"{src.stem}.tex"
        texf.write_text(tex, encoding="utf-8")
        warn, err, pages = [], [], None
        for _ in range(2):                       # twice for longtable pagination
            p = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory", str(work), str(texf)],
                capture_output=True, text=True, cwd=str(rel_root))
            log = p.stdout + p.stderr
        for m in re.finditer(r"^(LaTeX Warning|Overfull|Underfull).*$", log, re.M):
            warn.append(m.group(0)[:110])
        for m in re.finditer(r"^!.*$", log, re.M):
            err.append(m.group(0)[:110])
        pdf = work / f"{src.stem}.pdf"
        if pdf.exists():
            mm = re.search(r"Output written on .*?\((\d+) pages?", log)
            pages = int(mm.group(1)) if mm else None
            shutil.copy2(pdf, gen / f"{src.stem}.pdf")
        results.append({"part": part, "source": src.name,
                        "pdf": f"{src.stem}.pdf" if pdf.exists() else None,
                        "pages": pages, "n_warnings": len(warn),
                        "n_errors": len(err), "errors": err[:3],
                        "built": pdf.exists()})
        print(f"  {src.stem:44} {'OK' if pdf.exists() else 'FAILED':6} "
              f"pages={pages}  warn={len(warn)}  err={len(err)}")
        if err:
            for e in err[:3]:
                print(f"      ! {e}")
    return results
