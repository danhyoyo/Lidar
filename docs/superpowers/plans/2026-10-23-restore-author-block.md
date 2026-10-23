# Restore Author Block and Running Headers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the complete author list (`Duyen Huu Nguyen`, `Anh Duc Bui`, `Nhan Trong Phan`, `Hoang Ngoc Tran*`), institutional affiliations, and running author header in `paper/MobilePIXORNeXt.tex` while ensuring 0 formatting shifts across the 27-page manuscript.

**Architecture:** Springer `svjour3` twocolumn macro structure where `\author{...}` defines bold centered names separated by `\and`, `\authorrunning{...}` defines the even-page running header, and `\institute{...}` formats affiliations and corresponding author emails in the page 1 footer. An optional toggle `\newif\ifanonymous` can be included to cleanly support both single-blind (journal standard) and double-blind modes if required in the future.

**Tech Stack:** LaTeX2e (`svjour3.cls`, pdfTeX 3.141592653, TeX Live 2025/Debian), Poppler utilities (`pdftoppm`, `pdftotext`, `pdfinfo`).

**Spec:** User inquiry on missing author names under the title and editorial requirements for the Arabian Journal for Science and Engineering (AJSE), Springer Nature.

## Global Constraints
- Preserve total document length at exactly 27 pages with 0 Overfull `\hbox`.
- Maintain identical page transitions and float ordering across all 27 pages.
- Ensure the Overleaf ZIP package (`paper/MobilePIXORNeXt_Overleaf_Package.zip`) compiles cleanly with exit code 0.

---

### Root Cause Analysis: Why Author Names Were Missing

In Git commit `1d1059a`, line 27 of `paper/MobilePIXORNeXt.tex` was changed from the author list to `\author{}` (empty):
```latex
% [Audit-Action-Required]: Author block kept anonymous for double-blind submission gate verification.
% Author names for single-blind or camera-ready submission to AJSE:
% Duyen Huu Nguyen, Anh Duc Bui, Nhan Trong Phan, Hoang Ngoc Tran (FPT University)
\author{}
```
- **Reason:** A generic pre-submission audit script (`audit.py`) flagged `[FAIL] Anonymous submission (blind review check)` when author names were detected. The author field was temporarily emptied to pass the automated gate check.
- **Journal Reality:** The Arabian Journal for Science and Engineering (AJSE, Springer Nature) uses standard single-blind peer review for initial submissions. Author names, affiliations, and corresponding author details are required on the manuscript title page. Blanking `\author{}` left the title banner without authors while leaving the affiliation in the footer detached from any author names.

---

## Proposed Changes

### File: `paper/MobilePIXORNeXt.tex`

```latex
<<<<
% [Audit-Action-Required]: Author block kept anonymous for double-blind submission gate verification.
% Author names for single-blind or camera-ready submission to AJSE:
% Duyen Huu Nguyen, Anh Duc Bui, Nhan Trong Phan, Hoang Ngoc Tran (FPT University)
\author{}

%\authorrunning{Short form of author list} % if too long for running head

\institute{  *Correspondence: Hoang Ngoc Tran \\
 \at $^1$ 
              FPT University (FPT), Can Tho, 94000, VietNam. \\
%  \at $^2$ 
%               Department of Mechatronic Engineering, Faculty of Mechanical
%             Engineering, Ho Chi Minh City University of Technology (HCMUT),
% Vietnam National University, Ho Chi Minh City, Vietnam. \\
              \email{\raggedright hoang2531992@gmail.com; duyennhce200017@gmail.com; anhbdce200052@gmail.com; nhanptce200090@gmail.com}           %  \\
%             \emph{Present address:} of F. Author  %  if needed
}
====
\author{
        Duyen Huu Nguyen$^1$ \and
        Anh Duc Bui$^1$ \and
        Nhan Trong Phan$^1$ \and
        Hoang Ngoc Tran$^{*,1}$
}

\authorrunning{D. H. Nguyen et al.}

\institute{
  $^*$Corresponding author: Hoang Ngoc Tran (\email{hoang2531992@gmail.com}) \\
  \at $^1$ FPT University (FPT), Can Tho, 94000, VietNam. \\
  \email{duyennhce200017@gmail.com; anhbdce200052@gmail.com; nhanptce200090@gmail.com; hoang2531992@gmail.com}
}
>>>>
```

---

## Visual Verification & Layout Safety

A trial compilation with the proposed author block confirmed:
1. **Title Banner**: Author names display properly below the title with standard dot separators:
   `Duyen Huu Nguyen¹ · Anh Duc Bui¹ · Nhan Trong Phan¹ · Hoang Ngoc Tran*¹`
2. **Running Headers**: Even-numbered pages display `D. H. Nguyen et al.` (right header) and odd-numbered pages display `MobilePIXORNeXt` (left header).
3. **Footer**: Cleanly renders the correspondence note and FPT University email list.
4. **Zero Page Displacement**: The vertical height allocated to the title block in `svjour3` accommodates the author line without pushing text. Page 1 ends on the exact same line (`high throughput while preserving geometric precision.`), and all subsequent page boundaries (Pages 1 to 27) remain identical.

---

## Tasks

### Task 1: Update Author and Running Headers in Manuscript
- [ ] Edit `paper/MobilePIXORNeXt.tex` to populate `\author{...}`, `\authorrunning{...}`, and clean up `\institute{...}`.
- [ ] Recompile PDF using `compile.py --recipe pdflatex-bibtex`.
- [ ] Verify compilation exit code is 0 and 0 Overfull `\hbox` warnings.

### Task 2: Visual and Text Verification
- [ ] Extract Page 1 text via `pdftotext` to confirm author names and affiliations.
- [ ] Render Page 1 to PNG via `pdftoppm` to inspect visual formatting.
- [ ] Confirm total page count remains exactly 27 pages with identical page boundaries.

### Task 3: Packaging and Audit
- [ ] Run quality gate audit.
- [ ] Rebuild and test `paper/MobilePIXORNeXt_Overleaf_Package.zip` in an isolated directory.
- [ ] Update `walkthrough.md`.
