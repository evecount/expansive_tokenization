# 🏛️ Expansive Tokenization: Context-of-Interest Vocabulary & Semantic Shift Workbook

> **Tokenization built upon context of interest** — An interactive technical & sociocultural workbook for designing, expanding, and evaluating **Persona-, Dialect-, and Domain-Specific Tokenization Sets** and their downstream impact on **Contextual Token Embeddings**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/expansive_tokenization/blob/main/Expansive_Tokenization_Workbook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Tokenizers%20%26%20Embeddings-ffb000.svg)](https://huggingface.co/docs/transformers)

---

## 💡 The Core Premise: Why "One-Size-Fits-All" Tokenization Breaks Meaning

Most modern language models rely on a **static, universal subword vocabulary** (such as Byte-Pair Encoding or WordPiece trained on generic web crawls). While convenient, a static tokenizer makes a brittle assumption: **that every speaker, community, dialect, and domain carves language into the exact same atomic units.**

In reality, **tokenization is an act of classification based on who is speaking, who is listening, and what cultural or technical world they inhabit:**

1. **Subword Fragmentation Destroys Semantic Atomicity:**
   When a general-purpose tokenizer encounters domain-specific, mythical, or dialectal terms—like `katabasis` (Greek mythology), `dépanneur` (Québécois French), or `skibidi` (Gen-Z slang)—it shatters them into arbitrary, meaningless character shards (`["kat", "##aba", "##sis"]`). The model is forced to spend attention heads reassembling spelling rather than reasoning over meaning.
2. **Polysemy & Interlocutor-Dependent Embeddings:**
   Even when the exact same surface word is used—e.g., *"bet"*, *"cooked"*, *"tea"*, *"gradient"*, *"boot"*, or *"char"*—its vector representation in 768-dimensional embedding space shifts dramatically depending on the surrounding **context of interest** and the **interlocutor profile**.
3. **The Future of Adaptive Language Models:**
   Rather than forcing every human into a flattened global vocabulary, future cognitive systems will dynamically select or expand **Context-of-Interest Tokenization Sets** tailored to the person and domain they are engaging with.

---

## 🧭 The 8 Archetypal Context-of-Interest Packs Included in This Workbook

This repository functions as a **Tokenization Set Generator & Semantic Shift Workbook** for any human passion, fandom, literary canon, or regional dialect. It includes both a **runnable Jupyter Workbook (`Expansive_Tokenization_Workbook.ipynb`)** and a **modular Python toolkit (`expansive_tokenization/`)** equipped with eight pre-built tokenization & semantic contrast packs, plus an interactive generator to craft your own for *anything* (from LARPing and fandoms to specialized academic disciplines):

| # | Context-of-Interest Pack | Example Atomic Vocabulary / Idioms | What It Demonstrates |
| :- | :--- | :--- | :--- |
| **1** | 🏛️ **Greek Mythology & Epic Poetry** | `katabasis`, `kleos`, `nostos`, `xenia`, `hubris`, `aegis`, `rhapsode`, `dactylic_hexameter`, `chthonic`, `apotheosis` | How classical & poetic concepts fracture under BPE vs. preserving compound epithets (`wine-dark_sea`, `rosy-fingered_dawn`) as single semantic tokens. |
| **2** | 🏺 **Egyptology, Edgar Allan Poe Gothic & LARP / Fandoms** | **Egyptology:** `ma'at`, `isfet`, `ka_and_ba`, `duat`, `cartouche`, `ushabti`<br>**Poe Gothic:** `nevermore`, `tintinnabulation`, `amontillado`, `sepulchre`<br>**LARP / Fandom:** `in-character`, `phys-rep`, `bleed_effect`, `headcanon`, `lore_drop` | How a **Tokenization Set Generator** can plug into hyper-specific human interests, literary classics (e.g., the complete works of Edgar Allan Poe), ancient history, or live-action roleplaying (LARP) & fandom worldbuilding. |
| **3** | 🧬 **General Speaker vs. ML Researcher** | `attention_head`, `token_embedding`, `gradient_descent`, `backpropagation`, `lora_adapter`, `kv_cache`, `subword_fertility` | How the exact same words (*"attention"*, *"weights"*, *"loss"*) occupy completely different 768-D embedding neighborhoods when speaking to a layperson vs. an ML engineer. |
| **4** | ⚡ **Generational Slang: Millennial vs. Gen-Z** | **Millennial:** `adulting`, `doggo`, `on_fleek`, `side_hustle`, `FOMO`<br>**Gen-Z:** `no_cap`, `rizz`, `skibidi`, `delulu`, `main_character_energy`, `ate_and_left_no_crumbs` | Cross-generational semantic drift: how shared English tokens like *"cooked"*, *"bet"*, *"ate"*, *"aura"*, and *"based"* invert or pivot their vector geometry across cohorts. |
| **5** | 🌍 **World Englishes & Commonwealth Nuances** | **SG/MY:** `shiok`, `kiasu`, `lepak`, `makan`, `tapau`<br>**UK/AU/CA:** `fortnight`, `chuffed`, `arvo`, `fair_dinkum`, `toque`, `double-double` | Regional lexical anchors and divergence of shared nouns (*"boot"*, *"scheme"*, *"flat"*, *"pavement"*) across Commonwealth varieties vs. US English defaults. |
| **6** | ⚜️ **Francophone Diatopic Nuances (Québec vs. Paris Region vs. Francophonie)** | **Québec:** `char`, `dépanneur`, `chum`, `blonde`, `tiguidou`, `poudrerie`, `magasiner`<br>**Paris/Verlan:** `meuf`, `chelou`, `relou`, `boulot`<br>**West Africa:** `griot`, `teranga`, `maquis` | How a single language (French) requires distinct tokenization sets and contextual embeddings across Québec, regional Parisian registers, and broader Francophone territories. |
| **7** | 🧭 **Iberian, Lusophone & Germanic Regional Continuums** | **PT-PT vs. PT-BR:** `autocarro` / `ônibus`, `pequeno-almoço` / `café_da_manhã`, `telemóvel` / `celular`<br>**Castilian & Catalan:** `sobremesa`, `seny`, `rauxa`<br>**Regional UK & Germanic:** `canny`, `bairn`, `scran`, `ginnel`, `Fingerspitzengefühl` | Dialectal & diatopic parallels across European vs. Brazilian Portuguese (*"rapariga"*), Spanish vs. Catalan, and regional English tonalities (Geordie/Scouse/Yorkshire) with Germanic roots. |
| **8** | 🦃 **Sensory & Culinary Scene Decomposition ("The Turkey Dinner Experiment")** | `roast_turkey`, `mashed_potatoes`, `roasted_pumpkin`, `cranberry_sauce`, `pan_gravy`, `convection_oven`, `warmed_ceramic_plates`, `maillard_crust` | Tokenizing a physical/sensory scene into compositional ontological slots (`[PROTEIN]`, `[STARCH]`, `[ACID_GARNISH]`, `[VESSEL]`, `[HEAT_SOURCE]`) rather than raw character sequences. |

---

## 📂 Repository Structure

```text
expansive_tokenization/
├── Expansive_Tokenization_Workbook.ipynb                      # 🎯 Main Interactive Workbook (7 Guided Labs + Custom Set Builder)
├── GeneratingFirstText_Tokenization_TokenEmbedding.ipynb      # 📚 Foundational Reference Notebook (Phi-3 Generation, BPE & BERT 768-D Basics)
├── index.html                                                 # 🌐 Interactive Browser Studio (Test & Build Tokenization Sets Live!)
├── src/
│   └── expansive_tokenization/
│       ├── __init__.py
│       ├── presets.py                                         # Curated Context-of-Interest Token Sets & Polysemy Contrast Corpora
│       └── engine.py                                          # Vocabulary Expansion, Fertility/Fragmentation Audit & Embedding Shift Engine
├── requirements.txt                                           # Python dependencies (transformers, torch, scikit-learn, matplotlib, pandas)
└── README.md                                                  # Documentation & Quickstart Guide
```

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/evecount/expansive_tokenization.git
cd expansive_tokenization
pip install -r requirements.txt
```

### 2. Launch the Interactive Workbook

Open **`Expansive_Tokenization_Workbook.ipynb`** in JupyterLab, VS Code, or Google Colab:

```bash
jupyter notebook Expansive_Tokenization_Workbook.ipynb
```

Or open **`index.html`** directly in your browser to use the **Interactive Expansive Tokenization Studio** with zero setup!

---

## 🧪 What You Will Build in the Workbook

1. **Lab 1 — Baseline Tokenization & The Subword Fragmentation Audit:**
   Load standard Hugging Face tokenizers (`bert-base-uncased`, `gpt2`, or `microsoft/Phi-3-mini-4k-instruct`) and measure **Subword Fertility** (tokens per word) and **Semantic Fracture Rate** across specialized domains.
2. **Lab 2 — Designing Domain & Scene Tokenization Sets (Greek Myth & The Turkey Dinner Experiment):**
   Construct structured `TokenizationSet` schemas with domain vocabulary, multi-word expressions (MWEs), and ontological slot tags.
3. **Lab 3 — Expanding a Live Tokenizer & Resizing Model Embeddings:**
   Programmatically inject your custom token set into a Hugging Face `AutoTokenizer`, resize the model's token embedding matrix (`model.resize_token_embeddings`), and initialize new token vectors via **Compositional Semantic Averaging** so the model immediately understands them without full retraining.
4. **Lab 4 — Contextual Embedding Geometry (768-D Vector Shifts Across Contexts):**
   Prove that token embeddings are dynamic classifications of sentence relationships. Compute cosine similarity matrices showing how identical words shift in 768-dimensional BERT space when spoken by a **General Speaker vs. ML Engineer**, **Millennial vs. Gen-Z**, or in **Québec vs. Paris French**.
5. **Lab 5 — The Custom Tokenization Set Builder & Exporter:**
   Use the interactive Python template to author your own domain, fandom, dialect, or persona tokenization set, audit its compression & semantic coherence gains, and export a ready-to-share JSON/HuggingFace artifact.

---

## ✍️ Authors & Philosophy

Created by **Eve Count (`evecount`)** — exploring how language models perceive human nuance, cultural identity, and domain expertise through context-aware tokenization and representation geometry.
