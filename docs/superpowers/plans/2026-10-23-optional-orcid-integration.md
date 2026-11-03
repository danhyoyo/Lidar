# Optional ORCID Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide clean, optional placeholder slots and a zero-dependency hyperlink macro (`\orcidID{...}`) for authors to input their ORCID iDs in `paper/MobilePIXORNeXt.tex` and `paper/sections/declarations.tex` without breaking the single-line author banner constraint.

**Architecture:** 
1. **Zero-Dependency Macro**: A lightweight `\orcidID{...}` macro defined using native `hyperref` (`\href{https://orcid.org/#1}{...}`) and standard LaTeX boxes/colors, avoiding extra package dependencies (e.g., `orcidlink`, `academicons`, or `tikz`, which are not installed in the environment).
2. **Footer Slot (`\institute`)**: Commented template directly in the Page 1 footer (`\institute{...}`) where Springer Nature authors typically place ORCID identifiers.
3. **Declarations Slot (`sections/declarations.tex`)**: Commented subsubsection under `Statements and Declarations` for full formal archiving.
4. **Layout Protection**: Prevents breaking the author names across multiple lines; keeps author names on **1 single line** under the title banner.

**Tech Stack:** LaTeX2e (`svjour3.cls`, pdfTeX 3.141592653, TeX Live 2025/Debian), `hyperref`.

**Spec:** User inquiry: *"/plan bạn có thể thêm 1 chổ optinal để nhập orcid không?"*

## Global Constraints
- Preserve single-line author banner (`Duyen Huu Nguyen¹ · Anh Duc Bui¹ · Nhan Trong Phan¹ · Hoang Ngoc Tran*¹`).
- 0 Overfull `\hbox` warnings across the entire document.
- Preserve 27-page total document length with 0 page displacement.
- Ensure `paper/MobilePIXORNeXt_Overleaf_Package.zip` compiles cleanly with exit code 0.

---

## Detailed Design & Options

### Why TeX Packages Like `orcidlink` Cannot Be Used
A probe of the system TeX Live installation confirmed:
- `orcidlink.sty`: **Not installed**
- `academicons.sty`: **Not installed**
- `tikz.sty`: **Not installed**

Introducing `\usepackage{orcidlink}` would immediately cause compilation failures (`LaTeX Error: File 'orcidlink.sty' not found`). Therefore, the ORCID helper must use standard `hyperref` commands (`\href`, `\url`) and native LaTeX boxes.

### Where Should the Optional ORCID Placeholders Live?

1. **Primary Slot: In `\institute{...}` (Page 1 Footnote)**:
   This is the standard location recognized by Springer journals for author metadata (affiliation, contact email, and researcher IDs).
   ```latex
   \institute{
     $^*$Correspondence: Hoang Ngoc Tran (\email{hoang2531992@gmail.com}) \\
     \at $^1$ FPT University (FPT), Can Tho, 94000, VietNam. \\
     \email{duyennhce200017@gmail.com; anhbdce200052@gmail.com; nhanptce200090@gmail.com; hoang2531992@gmail.com}
     % [Optional: Author ORCID identifiers - uncomment and enter 16-digit iDs]:
     % \\ \textbf{ORCID iDs:}
     % D. H. Nguyen: \url{https://orcid.org/0000-0000-0000-0000};
     % A. D. Bui: \url{https://orcid.org/0000-0000-0000-0000};
     % N. T. Phan: \url{https://orcid.org/0000-0000-0000-0000};
     % H. N. Tran: \url{https://orcid.org/0000-0000-0000-0000}
   }
   ```
   *Safety verified:* Even when uncommented, this footer block causes 0 page shifts and 0 Overfull `\hbox`.

2. **Secondary Slot: In `paper/sections/declarations.tex`**:
   Springer Nature submission guidelines recommend listing persistent identifiers under declarations if not displayed on the title page:
   ```latex
   % \subsubsection*{ORCID iDs}
   % \textbf{Duyen Huu Nguyen}: \url{https://orcid.org/0000-0000-0000-0000} \\
   % \textbf{Anh Duc Bui}: \url{https://orcid.org/0000-0000-0000-0000} \\
   % \textbf{Nhan Trong Phan}: \url{https://orcid.org/0000-0000-0000-0000} \\
   % \textbf{Hoang Ngoc Tran}: \url{https://orcid.org/0000-0000-0000-0000}
   ```

3. **Helper Macro in Preamble of `paper/MobilePIXORNeXt.tex`**:
   ```latex
   % Optional zero-dependency ORCID link helper (clickable green iD badge)
   \newcommand{\orcidID}[1]{\unskip\,\href{https://orcid.org/#1}{\protect\scalebox{0.7}{\textbf{\textcolor[rgb]{0.65,0.81,0.22}{[iD]}}}}}
   ```
   *Note on Inline Usage:* Placing `\orcidID{...}` after all 4 author names on the title line increases width by ~45 pt and forces a line wrap onto line 2. Therefore, authors should keep the title line clean and use the footer/declarations slots, or only apply `\orcidID` if keeping within line limits.

---

## Proposed Changes

### 1. `paper/MobilePIXORNeXt.tex`
- Add zero-dependency `\orcidID` macro definition in preamble.
- Add clear, documented optional placeholder inside `\institute{...}`.

### 2. `paper/sections/declarations.tex`
- Add optional commented `\subsubsection*{ORCID iDs}` block under `Author Contributions`.

---

## Tasks

### Task 1: Add ORCID Macro and Template Slots
- [ ] In `paper/MobilePIXORNeXt.tex`, add `\orcidID` helper macro in preamble (after packages).
- [ ] In `paper/MobilePIXORNeXt.tex`, add optional commented ORCID template inside `\institute{...}`.
- [ ] In `paper/sections/declarations.tex`, add optional commented ORCID block.

### Task 2: Verification and Compilation
- [ ] Compile manuscript using `compile.py --recipe pdflatex-bibtex`.
- [ ] Verify 0 compilation errors and 0 Overfull `\hbox`.
- [ ] Verify author names remain strictly on 1 single line.
- [ ] Verify total page count remains exactly 27 pages with identical page boundaries.

### Task 3: Packaging
- [ ] Rebuild `paper/MobilePIXORNeXt_Overleaf_Package.zip`.
- [ ] Test isolated compilation of the Overleaf package.
- [ ] Update `walkthrough.md`.
