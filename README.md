# AI-Based Classification Pipeline for Structural Engineering Literature

This repository contains a Python-based classification pipeline for
large-scale, reproducible classification of academic literature in
structural engineering.

The code was developed to support systematic mapping and review studies
of AI applications in finite-element–based structural analysis, design,
and automation. It processes BibTeX files and associated PDFs, enriches
each entry with structured classification fields, and exports summary
statistics for downstream analysis.

---

## Key features

- Automated processing of BibTeX + PDF corpora
- Token-aware PDF ingestion for large documents
- LLM-based multi-label classification
- Dynamic parallelism with token-per-minute (TPM) monitoring
- In-place enrichment of BibTeX entries
- Per-folder and global CSV summaries
- Scales to thousands of papers

## Related publication

This repository supports the methodology presented in:

> Vaktskjold, V. E., Toppe, L. O., Luczkowski, M., Rønnquist, A., Morin, D.  
> **Systematic Mapping of Artificial Intelligence Applications in Finite-Element-Based Structural Engineering**  
> *Buildings*, 2026.  
> https://www.mdpi.com/2075-5309/16/3/644

If you use this code in academic work, please cite the paper.
