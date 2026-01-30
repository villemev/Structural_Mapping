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
