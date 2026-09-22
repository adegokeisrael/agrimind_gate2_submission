# Dataset Provenance

## Sources

### 1. FarmerChat (Nigeria subset)
- **Source:** `DigiGreen/farmerchat-queries-large` (Hugging Face)
- **License:** CC-BY 4.0 — attribution to Digital Green required
- **Filtering:** Filtered to `user_country == "Nigeria"` (296,671 records), then records containing the `[REDACTED]` placeholder were excluded as a conservative privacy/data-hygiene measure (258,491 excluded), retaining 38,180 records (12.87% of the Nigerian subset).
- **Fields retained:** `query`, `response` only.

### 2. KisanVaani Agriculture QA
- **Source:** `KisanVaani/agriculture-qa-english-only` (Hugging Face)
- **License:** Apache 2.0
- **Size:** ~22,615 records
- **Fields:** `question`/`answer`, renamed to `query`/`response`.

## Cleaning Pipeline

1. Nigeria-filter FarmerChat by `user_country`.
2. Exclude records containing `[REDACTED]` (not reconstructed — excluded conservatively).
3. Retain only `query`/`response` fields from both sources.
4. Clean response text: remove personalized greetings, `[REDACTED]` remnants, generic conversational openings, unnecessary Markdown, excessive whitespace.
5. Remove empty query/response pairs.
6. Remove exact-duplicate pairs within each source, then across sources after concatenation.
7. Standardize both sources to a unified `query,response` schema.

## Privacy Considerations

Records containing `[REDACTED]` were excluded rather than reconstructed. Personalized greetings/names were stripped. Geographic context was retained where present — not treated as personally identifying, and agronomically relevant.

## Checksums

<!-- REPLACE — run against your actual final files:
     sha256sum model-Q4_K_M.gguf
     sha256sum adapter_model.safetensors   (if included below)
-->

| File | SHA256 |
|---|---|
| Final quantized GGUF (model-Q4_K_M.gguf) | `[TO BE FILLED]` |
| LoRA adapter (if included) | `[TO BE FILLED]` |

## Training Notebook

<!-- REPLACE — paste shareable Kaggle notebook link(s) -->
- SFT training notebook: `[TO BE FILLED]`
- Context-length fix + verification notebook: `[TO BE FILLED]`

## Merge & Quantization

LoRA adapter merged via `merge_and_unload()`, converted to GGUF via `convert_hf_to_gguf.py`, quantized to Q4_K_M via `llama-quantize`, then context-length metadata corrected via `gguf_set_metadata.py` (see REPORT.md Section 7).
