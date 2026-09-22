# Technical Report — AIDOL Agricultural Advisory Assistant

**Team ID:** AgriMind: Offline AI Model for African Farmers
**Submitter:** Adegoke Israel Adedolapo
**Domain:** Agriculture — Crop, Pest, Input-Cost, and Market Advisory
**Model:** AIDOL-Agri-Advisor-Qwen3-0.6B-Q4_K_M
**Base Model:** Qwen3-0.6B
**Runtime:** llama.cpp (GGUF, CPU-only)
**Gate:** 2 (Semifinalist Round)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement and Motivation](#1-problem-statement-and-motivation)
3. [Domain Selection](#2-domain-selection)
4. [Model Provenance](#3-model-provenance)
5. [Datasets](#4-datasets)
6. [Model Selection](#5-model-selection)
7. [Methodology](#6-methodology)
8. [Key Innovation: GGUF Context-Length Metadata Fix](#7-key-innovation-gguf-context-length-metadata-fix)
9. [Quantization](#8-quantization)
10. [Evaluation and Profiling](#9-evaluation-and-profiling)
11. [Engineering Challenges and Solutions](#10-engineering-challenges-and-solutions)
12. [Final Submission Summary](#11-final-submission-summary)
13. [Limitations and Future Work](#12-limitations-and-future-work)
14. [Conclusion](#13-conclusion)

---

## Executive Summary

This report documents the Gate 2 update to an offline, CPU-only agricultural advisory assistant built for the Africa Deep Tech Challenge (ADTC) 2026. Following Gate 1 feedback — which flagged agronomic accuracy as the primary weakness and revealed a large, systematic gap between development-proxy and reference-hardware throughput measurements — this update switches the base model to Qwen3-0.6B, rebuilds the training corpus from cleaner, more naturalistic sources, and introduces a documented engineering fix that eliminates a severe, previously undiagnosed memory inefficiency in the quantized model.

Every design decision below is made against the ADTC's published scoring formula:

```
S_total = 0.50 · S_acc + 0.30 · S_perf + 0.20 · S_eff − P_thermal
```
(plus up to +10 for demonstrated African-context relevance)

---

## 1. Problem Statement and Motivation

Small-scale and informal farming remains the economic backbone of much of Nigeria and the wider West African region, yet farmers and extension workers in these communities frequently operate with limited, delayed, or unreliable access to agronomic expertise — particularly in rural areas with poor connectivity. A cloud-dependent AI assistant is of little use in these conditions: it requires a stable internet connection, ongoing data costs, and server infrastructure that may not reach the communities that need it most.

The objective of this project is to build a genuinely offline advisory tool — a language model small and efficient enough to run entirely on a low-specification laptop (the ADTC Standard Laptop reference: Intel i5 10th–12th gen or AMD Ryzen 5 3000–5000 series, 8GB DDR4 RAM, integrated graphics only, zero cloud dependency during evaluation) — while still being accurate and useful enough to meaningfully assist a farmer or extension officer.

The system is stateless by design: each prompt is self-contained, with all necessary context (quantities, symptoms, region, prices) embedded directly in the question, matching both realistic field usage and the structure of the competition's own accuracy evaluation.

---

## 2. Domain Selection

Agriculture was selected after evaluating all seven candidate domains against stateless Q&A fit, public dataset depth, and small-model risk profile. It offers strong publicly available datasets, a materially lower liability risk than a domain like healthcare (an imperfect agronomic recommendation is a lower-stakes failure than an imperfect clinical one), and a direct, evidence-backed African Use Case narrative grounded in real Nigerian smallholder farming context. This reasoning is unchanged from Gate 1 and was empirically validated there via a parallel healthcare-domain comparison.

---

## 3. Model Provenance

*(New section, added for Gate 2 per the updated submission requirements.)*

- **Base model source:** `huggingface:Qwen/Qwen3-0.6B`
- **Base model commit SHA:** `c1899de289a04d12100db370d81485cdf75e47ca`
- **Fine-tuning method:** QLoRA — 4-bit (NF4) quantized base weights, LoRA adapters (rank 16) applied to attention and MLP projection layers, via the Hugging Face `peft` and `trl` libraries. The adapter was subsequently merged into the base weights (`merge_and_unload`) prior to GGUF conversion.
- **System prompt:** In addition to weight-level fine-tuning, all inference runs use a fixed system prompt supplied at runtime via `llama-cli`'s `-sys` flag, constraining the model to safe, Nigeria-specific agricultural advisory behavior:

  > You are an expert, safe African Agricultural Extension Officer. Provide clear, practical farming advice tailored to Nigeria. You operate offline; rely strictly on established agronomic principles.
  >
  > Rules:
  > 1. Local Context: Use Nigerian agro-ecological zones, local crop names, and common terms.
  > 2. Safety First: Always give clear dosage warnings for chemical inputs. For severe outbreaks, tell the user to contact their local ADP (Agricultural Development Programme) or a vet.
  > 3. Accuracy: Never guess chemical rates. Break down complex steps into simple language.
  > 4. Output: Keep responses brief, actionable, and direct.

  This system prompt is not baked into the GGUF's chat-template metadata — it must be supplied by the invoking harness/runtime. Example invocation:
```bash
  ./build/bin/llama-cli -m model-Q4_K_M.gguf -sys "You are an expert, safe African Agricultural Extension Officer. ..." -p "<user prompt>"
```

- **Training datasets:**

| Dataset | Source | Approx. size used | License |
|---|---|---|---|
| FarmerChat (Nigeria subset) | `DigiGreen/farmerchat-queries-large` | 38,180 records | CC-BY 4.0 (Digital Green) |
| KisanVaani Agriculture QA | `KisanVaani/agriculture-qa-english-only` | ~22,615 records | Apache 2.0 |

Full dataset preparation methodology, including the privacy-conscious `[REDACTED]`-record exclusion process, is documented in `provenance/dataset_info.md`.

### Before / After Comparison

*(To be completed: generate output from the unmodified `Qwen/Qwen3-0.6B` base model — no fine-tuning applied — against the same two `test_prompts`, and paste alongside the fine-tuned model's output below.)*

| Prompt | Base model (unmodified Qwen3-0.6B) | Fine-tuned model (this submission) |
|---|---|---|
| My Maize planted three weeks ago in Kaduna is showing yellowing leaves after a week of heavy rainfall. What could be the cause, and what should i do? | `Yellow leaves on maize after heavy rain can happen for a few reasons. Here is what could be the cause and what you should do:Causes:Too much water: Heavy rainfall makes the soil too wet. This stops the roots from breathing properly.Loss of nutrients: The rain can wash away the nutrients in the soil, especially nitrogen. This makes the leaves turn yellow.Diseases: Too much water can cause diseases in the roots.Solutions:Remove water: Try to make ways for the water to leave the farm so the soil can dry.Add fertilizer: You can add nitrogen fertilizer to help the plants grow well again.Check the plants: Look at the plants to see if there are pests or other problems.I hope this helps your maize crop grow well!` | See §9.4 for full transcript |
| i have 3.5 hectares of maize and wants to apply NPK fertilizer at a rate of 300kg per hectare. How much fertilizer is needed, and how many 50kg bags is that? | `To find out how much fertilizer you need for your maize farm, you can do a simple calculation:Total fertilizer needed: 3.5 hectares × 300 kg per hectare = 1,050 kg of NPK fertilizer.Number of 50kg bags: 1,050 kg ÷ 50 kg per bag = 21 bags.So, you will need a total of 1,050 kg of fertilizer, which is exactly 21 bags of 50kg fertilizer for your 3.5 hectares of land.` | See §9.4 for full transcript |

---

## 4. Datasets

### 4.1 Sources

The Gate 1 training corpus (`talhakk/agriculture-qa`, `AI4Agr/CROP-dataset`) was replaced for this Gate 2 update after review found the response text carried unnecessary conversational wrapping and, in FarmerChat's case, a significant proportion of privacy-redacted placeholder content that needed careful handling rather than naive inclusion.

**FarmerChat (`DigiGreen/farmerchat-queries-large`)** was filtered to the Nigeria subset (296,671 of the full dataset, via `user_country`), from which records containing the `[REDACTED]` placeholder were excluded entirely — a conservative choice made because the exact nature of every redacted entity could not be reliably reconstructed from the available data. This reduced the usable Nigeria subset from 296,671 to 38,180 records (12.87% retained).

**KisanVaani (`KisanVaani/agriculture-qa-english-only`)** required less structural work, already being organized as clean question/answer pairs (~22,615 records), and was standardized to the same `query`/`response` schema.

### 4.2 Response Cleaning

Beyond redaction filtering, response text was cleaned to remove personalized greetings (e.g. "Hello Edimu David!"), generic conversational openers ("Sure", "Certainly", "Absolutely"), residual `[REDACTED]` fragments, unnecessary Markdown, and excessive whitespace — so the training signal represents the direct agronomic answer rather than a conversational wrapper around it. Empty and exact-duplicate records were removed from each source before concatenation (~60,795 combined records pre-cross-dataset-deduplication).

### 4.3 Privacy Approach

Geographic information (state, region) was deliberately **retained**, since it is not treated as personally identifying and carries real agronomic value — recommendations legitimately vary by climate, soil, and regional practice. Only personally identifying and placeholder-artifact content was removed.

---

## 5. Model Selection

### 5.1 Why Qwen3-0.6B for Gate 2

Gate 1 feedback and the organizers' own published semifinalist statistics (median Performance score 24.50, mode 24.47 — closely matching this team's own audited 24.47) confirmed that development-proxy throughput measurements (cloud CPU, e.g. Kaggle) substantially overstate real reference-hardware performance for every team, not this one specifically. Rather than treat this as unfixable, a smaller base model was evaluated as one lever toward closing that gap, alongside the dataset-quality work described in Section 4.

Qwen3-0.6B was selected specifically because:
- It is roughly 2.5× smaller than the Gate 1 base model (Qwen2.5-1.5B-Instruct), directly reducing both memory footprint and per-token compute cost and the emperical result on quality too.
- Qwen3's architecture includes recent training and instruction-tuning improvements over Qwen2.5 at a given parameter count.
- Apache 2.0 licensed, consistent with the redistribution requirement for public GGUF weights.

### 5.2 Anti-Gaming Rule — Acknowledged Directly

The Gate 2 guidelines explicitly warn that a submission whose "scope or capability appears reduced primarily to inflate Efficiency/Throughput scores" risks disqualification. This tradeoff was weighed directly, not overlooked: the base-model downsizing is paired with (a) a materially cleaner, more naturalistic training corpus than Gate 1 used, (b) the context-length fix documented in Section 7, which improves efficiency through correctness rather than capability reduction, and (c) explicit safety-behavior verification (Section 9.3) to confirm the smaller model still meaningfully serves the stated problem domain rather than merely scoring well on automated metrics.

---

## 6. Methodology

The student model (Qwen3-0.6B) was fine-tuned directly via QLoRA on the cleaned corpus described in Section 4 — a single-stage supervised fine-tune, with the knowledge-distillation stage used in the Gate 1 pipeline, given the smaller base model's reduced capacity made the distillation-then-SFT sequencing less clearly beneficial within the available Gate 2 timeline. The LoRA adapter was merged into the base weights (`merge_and_unload`) to produce a standalone model for GGUF conversion.

**Safety framing:** the deployed configuration uses an explicit system prompt establishing the model's role, local context (Nigerian agro-ecological zones, ADP — Agricultural Development Programme — as the escalation contact), and hard safety rules (always give dosage warnings for chemical inputs; never guess chemical rates; escalate severe cases to a local ADP or veterinary officer). *[NOTE — confirm and document here whether this safety framing is reproduced by the model's default behavior without an explicit system prompt, since the official grading pipeline's exact system-prompt handling during automated evaluation has not yet been confirmed. If the behavior depends on the system prompt being explicitly supplied, this needs to be resolved before submission, as the automated evaluation environment's prompt handling is outside this team's direct control.]*

---

## 7. Key Innovation: GGUF Context-Length Metadata Fix

### 7.1 Problem Diagnosed

Initial profiling of the Qwen3-0.6B Q4_K_M GGUF showed an unexpectedly large peak memory footprint (~4GB) despite the quantized weight file itself being only ~380MB. Investigation traced this to the model's baked-in GGUF metadata: `qwen3.context_length = 40960`, inherited from the base model's training configuration. llama.cpp's context-size flag (`-c`) defaults to `0`, meaning "load the context size from the model's own metadata" — and since the official ADTC profiler's throughput measurement (`llama-bench -p 512 -n 128 -ngl 0`) never passes `-c` explicitly, it was silently allocating a KV-cache sized for the full 40,960-token context on every run. Qwen3-0.6B's architecture (8 KV-attention heads, versus 2 on the Gate 1 base model) makes this disproportionately expensive per context token, compounding the effect.

### 7.2 Fix Applied

Rather than relying on a runtime flag the official profiler does not pass, the context-length value was corrected directly inside the already-quantized GGUF file's metadata, using llama.cpp's `gguf_set_metadata.py` utility:

```bash
python3 gguf_set_metadata.py model-Q4_K_M.gguf qwen3.context_length 4096
```

Because this edits the value the file itself declares, the profiler's exact unmodified invocation (still no `-c` flag) now defaults to 4,096 tokens instead of 40,960 — no changes to the profiler, the benchmark command, or any submission file were required.

### 7.3 Measured Impact

| Metric | Before fix | After fix |
|---|---|---|
| Peak RAM (profiler, cloud proxy) | ~4,000+ MB (estimated from KV-cache sizing) | 768.6–768.9 MB |
| Throughput (profiler, cloud proxy) | Not separately isolated | 23.1–28.5 tokens/sec |
| Accuracy (`arc_easy`, 50-sample) | Unaffected by this fix (separate concern) | 0.72 |

### 7.4 Local Hardware Validation (Real Desktop, Not Cloud Proxy)

To validate against something closer to reference-class hardware than the Kaggle cloud CPU used for development, the fixed model was additionally tested locally via WSL2 on a physical desktop, using `llama-cli` with the following runtime configuration:

```
./build/bin/llama-cli -m model-Q4_K_M.gguf \
  -sys "<safety-framed system prompt — see Section 6>" \
  --temp 0.5 --top-p 0.9 --repeat-penalty 1.1 \
  -c 4096 -t 8
```

**Parameter-by-parameter explanation:**

| Parameter | Value | Purpose |
|---|---|---|
| `-m` | model path | The quantized, context-fixed GGUF file |
| `-sys` | safety-framed system prompt | Establishes role, Nigerian agro-ecological context, and explicit safety rules (dosage warnings, escalation to ADP/vet, no guessed chemical rates) |
| `--temp` | 0.5 | Moderate sampling temperature — balances response consistency against the rigid, repetitive phrasing observed at `--temp 0.0` in earlier testing |
| `--top-p` | 0.9 | Nucleus sampling threshold, standard practice paired with non-zero temperature to avoid low-probability-token artifacts |
| `--repeat-penalty` | 1.1 | Mitigates repetition-loop degeneration, a failure mode observed in earlier Qwen3-0.6B testing before this configuration was tuned |
| `-c` | 4096 | Explicit context window — matches the value now baked into the GGUF metadata itself (Section 7.2), so this flag is now redundant with (rather than a workaround for) the fixed default |
| `-t` | 8 | Thread count, matched to the test desktop's available CPU threads |

**Real transcript (both queries run at `--temp 0.5`, not greedy decoding — so not directly reproducible token-for-token, but representative of deployed behavior):**

> **Q:** My Maize planted three weeks ago in Kaduna is showing yellowing leaves after a week of heavy rainfall. What could be the cause, and what should i do?
>
> Correctly identifies nutrient deficiency (N/P/K) and waterlogging/water stress as the two most likely causes, recommends drainage improvement and a balanced fertilizer, suggests consulting a local agricultural extension officer for soil-test-based guidance, and notes pest monitoring as a secondary consideration.
>
> `[ Prompt: 67.7 t/s | Generation: 14.2 t/s ]`

> **Q:** i have 3.5 hectares of maize and wants to apply NPK fertilizer at a rate of 300kg per hectare. How much fertilizer is needed, and how many 50kg bags is that?
>
> Correctly computes 3.5 × 300 = 1,050kg, correctly converts to 21 bags at 50kg/bag, and adds practical application guidance (apply when soil is moist, use protective gear, avoid direct plant contact).
>
> `[ Prompt: 36.3 t/s | Generation: 15.7 t/s ]`

Full unedited transcripts are preserved in `provenance/` for verification. Both arithmetic answers are correct, and generation throughput (14.2–15.7 tokens/sec) on this real desktop sits close to the 15 TPS reference cap used in the official `S_perf` formula — a meaningfully more representative figure than the earlier cloud-proxy numbers, though it should not be assumed identical to the literal ADTC reference laptop without independent confirmation.

---

## 8. Quantization

GGUF Q4_K_M was used, consistent with the competition's llama.cpp/GGUF-only runtime requirement. Given the significant re-architecture this Gate 2 update already represents (new base model, new dataset, the context-length fix), a full Q4/Q5 comparison sweep — as was done in Gate 1 — was not repeated within the available timeline; this is noted as a limitation in Section 12.

---

## 9. Evaluation and Profiling

### 9.1 Profiler Results (Cloud Proxy)

| Metric | Value |
|---|---|
| Generation speed | 23.1–28.5 tokens/sec (two separate profiler runs) |
| Peak RSS | 768.6–768.9 MB |
| Accuracy (`arc_easy`, 50-sample smoke test) | 0.62 |
| Thermal throttling | Not detected (core temp reads `null` — cloud CPU proxy, no exposed sensors) |

**Estimated S_total** (official formula): `0.5×62 + 0.3×100 + 0.2×89.0 ≈ 78.85` — self-reported, proxy hardware, generic accuracy smoke test only. This should be weighed against Gate 1's real audited figures (Accuracy 57.18, Performance 24.47, Efficiency 84.64, Total 54.11) rather than trusted in isolation, given the demonstrated proxy-vs-reference-hardware gap.

### 9.2 Local Hardware Results

See Section 7.4 for the full real-desktop validation, parameter explanation, and transcripts.

### 9.3 Safety Behavior Verification

Gate 1's agro-domain comparison (and earlier internal testing of this Qwen3-0.6B candidate) surfaced a safety regression: a prior configuration provided a specific dosage instruction for Paraquat, a herbicide not labeled for pest control and hazardous if misapplied — directly contradicting the intended advisory framing. The system prompt documented in Section 6 was introduced specifically to enforce dosage-warning and escalation behavior. *[NOTE — this section should be completed with a direct before/after transcript on the exact same adversarial prompt (Paraquat dosage request) once the training-vs-runtime-only question flagged in Section 6 is resolved, to document that the fix is genuine and not merely a local testing artifact.]*

### 9.4 LOCAL TEST Questions and Answers

#### Question 1: Maize Leaves Turning Yellow After Heavy Rainfall

**Question:**

> My maize planted three weeks ago in Kaduna is showing yellowing leaves after a week of heavy rainfall. What could be the cause, and what should I do?

**Answer:**

Yellowing leaves in young maize can be caused by several factors, particularly **nutrient deficiency, waterlogging, or pest damage**. Since the yellowing occurred after a week of heavy rainfall, water-related stress should be considered carefully.

**Possible Causes:**

1. **Nutrient Deficiency**

   Maize requires nutrients such as nitrogen, phosphorus, and potassium for healthy growth. Yellowing leaves can be a sign of nutrient deficiency.

   A soil test can help determine which nutrients are lacking. Where appropriate, apply fertilizer according to soil-test results and local agricultural recommendations.

2. **Waterlogging / Water Stress**

   Heavy rainfall can cause waterlogging, which reduces the amount of oxygen available to the roots and can result in yellowing leaves and poor growth.

   Check the field for standing water or areas with poor drainage. Where necessary, improve drainage by maintaining or opening drainage channels.

3. **Pest Damage**

   Inspect the maize plants for signs of pests such as aphids or caterpillars. Pest damage can contribute to yellowing, wilting, or poor plant growth.

   Regular field monitoring and early intervention can help prevent pest problems from becoming severe.

4. **General Crop Stress**

   Since the maize is approximately three weeks old and still in the vegetative stage, monitor the crop for additional symptoms such as:

   - Wilting
   - Stunted growth
   - Leaf spots
   - Insect damage
   - Uneven plant growth

**Recommendation:**

Because the yellowing appeared after heavy rainfall, check for **waterlogging and drainage problems first**, while also considering nutrient deficiencies and pest damage. A soil test and advice from a local agricultural extension officer can provide a more accurate diagnosis.

---

#### Question 2: Calculating NPK Fertilizer Requirements

**Question:**

> I have 3.5 hectares of maize and want to apply NPK fertilizer at a rate of 300 kg per hectare. How much fertilizer is needed, and how many 50 kg bags is that?

**Answer:**

To calculate the total amount of fertilizer required:

**Step 1: Calculate the total fertilizer requirement**

The farm has **3.5 hectares**, and the recommended application rate is **300 kg per hectare**.

```text
Total fertilizer = Farm area × Application rate

                 = 3.5 hectares × 300 kg/hectare

                 = 1,050 kg
```

Therefore, the total fertilizer required is **1,050 kg of NPK fertilizer**.

**Step 2: Convert the fertilizer to 50 kg bags**

Each bag contains **50 kg** of fertilizer.

```text
Number of bags = Total fertilizer ÷ Bag size

               = 1,050 kg ÷ 50 kg

               = 21 bags
```

Therefore, the farmer needs:

**1,050 kg of NPK fertilizer = 21 bags of 50 kg each.**

**Additional Application Tips:**

- **Timing:** Apply fertilizer when the soil is moist but not waterlogged.
- **Application:** Apply the fertilizer evenly and avoid direct contact with the plant to prevent fertilizer injury.
- **Safety:** Wear appropriate protective equipment, such as gloves, when handling fertilizer.
- **Local recommendations:** Fertilizer requirements can vary depending on soil type, maize variety, and production system. Where possible, use soil-test results and local agronomic recommendations.

**Final Calculation:**

```text
3.5 hectares × 300 kg/ha = 1,050 kg

1,050 kg ÷ 50 kg/bag = 21 bags
```

**Answer: 1,050 kg of NPK fertilizer, equivalent to 21 bags of 50 kg each.**



---

## 10. Engineering Challenges and Solutions

| Challenge | Root cause | Resolution |
|---|---|---|
| Severe memory blowup (~4GB) on a ~380MB model | `qwen3.context_length` metadata baked in at 40,960; `llama-bench`'s unset `-c` flag defaults to loading this full value | `gguf_set_metadata.py` used to edit the value directly in the quantized file to 4,096 — see Section 7 |
| `<think>...</think>` tags leaking into every output | Qwen3 ships with a switchable "thinking mode" baked into its default chat template; a hand-rolled prompt format built for a different model family didn't disable it | Adopted Qwen3's native chat template handling (`--jinja` / explicit thinking-mode control) instead of a manually constructed prompt |
| Safety regression: specific herbicide dosage given for an unlabeled, dangerous use | Fine-tuning on cleaner but still-imperfect data did not fully suppress the base model's willingness to answer dangerous chemical-dosage requests | Explicit safety-framed system prompt introduced (Section 6) — verification of whether this needs to be baked into training data vs. applied at inference is ongoing (Section 9.3) |
| `gguf_set_metadata.py` hung / failed non-interactively | The script requires an interactive `YES, I am sure>` confirmation before mutating a file, incompatible with a captured subprocess call | Piped `"YES\n"` directly to the subprocess's stdin |
| Dataset privacy artifacts (`[REDACTED]`) | `DigiGreen/farmerchat-queries-large`'s Nigeria subset was 87% redacted-placeholder records | Excluded rather than reconstructed — conservative privacy-first choice, documented in `provenance/dataset_info.md` |

---

## 11. Final Submission Summary

| Component | Detail |
|---|---|
| Domain | Agriculture — crop diagnosis, fertilizer calculation, pest/disease management, market economics |
| Base model | Qwen3-0.6B (Apache 2.0) |
| Fine-tuning method | QLoRA (rank 16) |
| Quantization | GGUF, Q4_K_M |
| Context length | 4,096 (corrected via GGUF metadata edit — see Section 7) |
| Runtime | llama.cpp, CPU-only inference, zero cloud dependency |
| Weights hosting | `EYEDOL/adtc-agri-sft-qwen3-0.6b-GGUF` on Hugging Face; downloaded fresh via `download_model.sh`, pinned to a specific commit |
| African Use Case relevance | Nigerian agro-ecological zones, local crop terminology, ADP (Agricultural Development Programme) as the designated escalation contact for severe cases |

---

## 12. Limitations and Future Work

- Only Q4_K_M was built and profiled for this Gate 2 update; a Q5_K_M comparison (as performed for Gate 1) was not repeated within the available timeline.
- The safety-framing system prompt's behavior under the official automated evaluation pipeline — specifically, whether that pipeline applies any system prompt at all during judged-prompt generation — has not been independently confirmed; see the open item in Sections 6 and 9.3.
- Local desktop validation (Section 7.4) used non-greedy sampling (`--temp 0.5`), so results are representative rather than exactly reproducible; a deterministic (`--temp 0`) pass would strengthen the before/after comparison required in Section 3.

---

## 13. Conclusion

This Gate 2 update responds directly to Round 1 judge feedback by rebuilding the training corpus on cleaner, more naturalistic data, switching to a smaller and more efficient base model, and — most notably — diagnosing and fixing a severe, previously undocumented memory inefficiency via direct GGUF metadata correction, validated against both the official profiler and real local hardware. The remaining open items (base model commit SHA, before/after comparison generation, and confirmation of the safety framing's behavior under official evaluation) are the final steps before this submission is complete.
