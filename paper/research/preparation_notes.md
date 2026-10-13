# Manuscript preparation record

Prepared on 2026-10-13 with the scientific-writing and pdf skills.

## Evidence and editorial decisions

The reference paper was read from the supplied PDF. The experiment workbook, 25-slide PowerPoint, Demo documentation, detector source, split manifests, 21 archived runs and 41 saved evaluations were examined. A separate trained Final3D run and its existing CPU diagnostics were incorporated after their identities and protocols were established.

Raw evaluation JSON takes precedence over spreadsheet formatting. Hist14, independent BEV heads, and the shared Final3D head are described from code and checkpoint/configuration evidence. One post-run head field was removed from copies of all 21 BEV configs to reproduce their recorded training SHA-256 values exactly. Both supplied and recovered copies remain in the package. The source experiment directories were not edited.

The main text separates local BEV, full-frame reference BEV/3D, and conditional CPU diagnostics. Claims of SOTA, repeated-seed significance, test-set accuracy, equal-recall false-positive improvements, and embedded-device performance are absent because the supplied evidence does not establish them. Demo cuboids with LiDAR-derived vertical extent are captioned accordingly.

The existing vector architecture counterpart is used for the main diagram. The original editable PowerPoint is preserved byte-for-byte, and its raster images and recursive text inventory are extracted. Early diagram labels are not used to override saved model specifications. Three new scientific plots are generated from saved numeric values and provided as PDF, SVG and 300-dpi PNG.

## Render and reference checks

Both documents were compiled with Tectonic 0.17.0 using IEEEtran/BibTeX for the main manuscript. Main: 8 pages; supplementary: 11 pages. All 23 BibTeX entries are cited across the two documents, with no missing or unused keys. No unresolved cross-reference, missing figure, source citation placeholder or overfull box remains. PDF font listings confirm all fonts are embedded. The main document retains minor underfull spacing warnings, mostly in long reference URLs; they do not represent cropped or overflowing content.

Page contact sheets and detailed pages covering the equations, provenance and tables were inspected visually. `document_checks.json` stores the final PDF digests and exact counts. `build_logs/` records compiler output. The 29-file source ZIP contains sources, bibliography, generated bibliography files, tables and figures used by the documents; it does not require the original training archives to compile.

## Revision: matched OGA--Q-OGA retraining (2026-10-13)

The user supplied a new archive at `/home/duyennh/Downloads/drive-download-20261013T095105Z-1-001` and requested an updated loss comparison. The preceding PDFs, manuscript source ZIP, notes, asset script and audits were archived as `revisions/v1_before_loss_retrain_20261013.zip` before changes.

The primary OGA--Q-OGA comparison now uses the matched O100/Q100 runs. They complete 100 epochs, use SimAM and independent grouped heads without IQA, and contain 815,744 detector parameters each. Their configurations differ only in `loss.name`; both already match the recorded training config digests. All four new raw reports match their checkpoint SHA-256 values, validation split, input asset hashes, encoding semantic digest and clean source revision. Epoch records establish completion and verify maximum scheduled AP and minimum validation-loss selection. Total BEV evidence is now 23 runs and 45 reports. Original received ablation CSVs remain evidence for the original 21 runs; combined JSON inventories, summaries and manuscript tables include the new pair.

At AP-selected epoch 70, Q-OGA scores 70.515097779% versus OGA's 69.690111393%, a 0.824986386 pp advantage calculated before rounding. Pedestrian contributes +4.210315976 pp; Car and Cyclist decrease by 0.110980489 and 1.624376328 pp. The loss-selected epochs 44 and 45 give a smaller 0.105202974 pp difference. OGA's loss-selected score is slightly higher than its scheduled-AP-selected score by 0.000801350 pp; epoch 45 is outside the AP schedule, so the primary selection policy is retained. Different raw objective definitions are not ranked as if their numerical scales were detector accuracy.

The original 50-epoch factorial and attention comparisons retain their corresponding results. The earlier P11 Q-OGA row remains historical supplementary evidence and is removed from the main factorial table and latency scatter. Abstract, Introduction, Experimental Setup, Results, Discussion and Conclusion integrate the new matched comparison. Supplementary material includes the exact scheduled AP table, loss-based checkpoint sensitivity, operating counts and recorded objective curves. Two additional plots are available as PDF/SVG/300-dpi PNG; the new evidence and derived comparison are under `research/loss_retrain/` and `research/loss_retrain_comparison.json`. The bibliography remains the same 23 verified cited sources.

Revised renders have 9 main pages and 13 supplementary pages, with all fonts embedded, no unresolved citations/references and no overfull boxes. The revised main document retains underfull spacing warnings. Page previews cover the updated tables, learning curves and provenance. `document_checks.json` and `revisions/v2_loss_retrain_20261013.json` record the final checks and artifact digests. The refreshed 34-file source ZIP includes every figure and table referenced by the revised documents. No detector experiment was run during this revision.

## Next author revision

Replace the anonymous author line and add venue-specific author/funding information. The English manuscript is complete as a draft grounded in the available experiments. Baselines, repeated seeds and matched 3D head experiments recommended in `venue_strategy_vi.md` remain future work. No new detector training, inference or detector test was performed during manuscript preparation.
