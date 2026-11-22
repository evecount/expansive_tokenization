# 🏛️ Expansive Tokenization: Context-of-Interest Vocabulary & Semantic Shift Workbook

> **Tokenization built upon context of interest** — An interactive technical & sociocultural workbook for designing, expanding, and evaluating **Persona-, Fandom-, Dialect-, and Domain-Specific Tokenization Sets** and their downstream impact on **Contextual Token Embeddings**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/expansive_tokenization/blob/main/Expansive_Tokenization_Workbook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Tokenizers%20%26%20Embeddings-ffb000.svg)](https://huggingface.co/docs/transformers)
[![Author: Gwendalynn Lim (SIT)](https://img.shields.io/badge/Author-Gwendalynn_Lim_%C2%B7_SIT-8b5cf6.svg)](#🎓-credits--academic-acknowledgments)

---

## 🧠 First Principles: What *Is* Tokenization? (A Baseline Understanding)

Before an AI language model can read a sentence, write a poem, or answer a question, it faces a fundamental hurdle: **neural networks cannot read raw text—they only process numbers and vectors.**

**Tokenization** is the very first translation layer of any Large Language Model (LLM). It is the process of taking human language and **classifying or carving it into meaningful building blocks called "tokens"**, then assigning each token a numerical ID and a high-dimensional coordinate (**a token embedding**, such as a 768-dimensional vector in BERT):

1. **Step 1 — Carving Text into Tokens (The Vocabulary):**
   When you type a sentence, an automatic tokenizer breaks it down into pieces. Common everyday words stay whole (`"cat"`, `"sat"`), while unfamiliar, specialized, or compound words get chopped into smaller subword fragments (`"apolog"` + `"izing"`).
2. **Step 2 — Projecting Tokens into Meaning Space (Token Embeddings):**
   Each token is mapped to a vector of numbers (e.g., 768 dimensions) that captures *what the token means in relation to the other words in the sentence*. In effect, **tokenization + embedding is a sophisticated act of classifying meaning based on how parts of a sentence relate to one another.**
3. **The Hidden Flaw in Today's "Universal" Tokenizers:**
   Almost all mainstream language models use a **frozen, one-size-fits-all vocabulary** generated from generic internet text. When you use a specialized term—whether from **Greek mythology** (`katabasis`), **anime fandoms** (`isekai`, `sakuga`), **Edgar Allan Poe** (`tintinnabulation`), or **Québécois French** (`dépanneur`)—a standard tokenizer doesn't recognize it as a whole concept. Instead, it shatters the word into meaningless syllables (`["tin", "##tin", "##na", "##bul", "##ation"]`).

---

## 🔮 The Premise: Why the Future of Tokenization Is *Not* Training Models from Scratch

A common misconception in AI is that if you want a model to deeply understand a specific community, dialect, academic discipline, or subculture, you have to **retrain a massive foundation model from scratch**.

**Training a model from scratch just to teach it a new cultural lens is completely redundant.** Foundation models already possess vast reasoning architectures. What they lack at inference or adaptation time is **the right lens of tokenization and contextual embedding routing for who they are speaking to right now.**

### **Tokenization of the Future = Dynamic Context-of-Interest Lenses**

Instead of a static dictionary that treats every human on Earth as an average web page, **the future of tokenization is adaptive, pluggable, and interlocutor-aware**:

* 🎭 **Reading Overt LARPing, Greek Mythology & Anime Fandoms:**
  If an AI knows it is talking to someone **overtly LARPing an Odyssey campaign**, analyzing **Greek Mythology & Epic Poetry**, or diving deep into **anime & manga lore**, it shouldn't shatter `katabasis`, `wine-dark_sea`, `phys-rep`, `bleed_effect`, `isekai`, `nakama`, or `sakuga` into broken character shards. By plugging in a **Context-of-Interest Tokenization Set**, the AI immediately treats those cultural artifacts as first-class atomic concepts and shifts its embedding interpretation to match that world.
* 🧬 **Adapting to the Interlocutor's Expertise (General Speaker vs. ML Researcher):**
  If an AI is chatting casually with a general user, words like *"attention"*, *"weights"*, and *"loss"* live in the everyday human neighborhood of focus, gym dumbbells, and grief. The moment the AI recognizes its interlocutor is studying **Machine Learning**, expanding the active token set (`attention_head`, `gradient_descent`, `lora_adapter`) primes the model to interpret those exact same surface words in linear-algebraic optimization space.
* 🌍 **Honoring Subtle Regional, Generational & Literary Nuances:**
  Whether distinguishing **Millennial vs. Gen-Z slang** (*"cooked"*, *"bet"*, *"ate"*), **Québécois French vs. Parisian French** (*"char"*, *"dépanneur"*), **European vs. Brazilian Portuguese** (*"autocarro"* vs. *"ônibus"*, *"rapariga"*), **Commonwealth & Regional British Englishes** (*"shiok"*, *"kiasu"*, *"chuffed"*, *"canny"*, *"scran"*), or **allegorical & thematic readings of literature and cinema** (e.g., a feminist reading of *Pride and Prejudice* vs. the sociopolitical commentary of *Django Unchained*), pluggable tokenization sets give AI the **subtlety of context** without redundant base-model retraining.

---

## 🧭 The 8 Archetypal Context-of-Interest Packs Included in This Workbook

This repository functions as an open-source **Tokenization Set Generator & Semantic Shift Workbook** for any human passion, fandom, literary canon, or regional dialect. It includes both a **runnable Jupyter Workbook ([Expansive_Tokenization_Workbook.ipynb](Expansive_Tokenization_Workbook.ipynb))** and a **modular Python toolkit (`src/expansive_tokenization/`)** equipped with eight pre-built tokenization & semantic contrast packs, plus an interactive generator to craft your own:

| # | Context-of-Interest Pack | Example Atomic Vocabulary / Idioms | What It Demonstrates |
| :- | :--- | :--- | :--- |
| **1** | 🏛️ **Greek Mythology & Epic Poetry** | `katabasis`, `kleos`, `nostos`, `xenia`, `hubris`, `aegis`, `rhapsode`, `dactylic_hexameter`, `chthonic`, `apotheosis` | How classical & poetic concepts fracture under BPE vs. preserving compound epithets (`wine-dark_sea`, `rosy-fingered_dawn`) as single semantic tokens. |
| **2** | 🏺 **Egyptology, Edgar Allan Poe Gothic, Anime & LARP / Fandoms** | **Egyptology:** `ma'at`, `isfet`, `ka_and_ba`, `duat`, `cartouche`, `ushabti`<br>**Poe Gothic:** `nevermore`, `tintinnabulation`, `amontillado`, `sepulchre`<br>**Anime & LARP:** `in-character`, `phys-rep`, `bleed_effect`, `headcanon`, `isekai`, `nakama`, `sakuga`, `chunibyo` | How a **Tokenization Set Generator** lets an AI immediately grasp overt LARPing, anime subcultures, ancient Egyptology, or the complete Gothic lexicon of Edgar Allan Poe without retraining from scratch. |
| **3** | 🧬 **General Speaker vs. ML Researcher** | `attention_head`, `token_embedding`, `gradient_descent`, `backpropagation`, `lora_adapter`, `kv_cache`, `subword_fertility` | How the exact same words (*"attention"*, *"weights"*, *"loss"*) occupy completely different 768-D embedding neighborhoods when speaking to a layperson vs. an ML engineer. |
| **4** | ⚡ **Generational Slang: Millennial vs. Gen-Z** | **Millennial:** `adulting`, `doggo`, `on_fleek`, `side_hustle`, `FOMO`<br>**Gen-Z:** `no_cap`, `rizz`, `skibidi`, `delulu`, `main_character_energy`, `ate_and_left_no_crumbs` | Cross-generational semantic drift: how shared English tokens like *"cooked"*, *"bet"*, *"ate"*, *"aura"*, and *"based"* invert or pivot their vector geometry across cohorts. |
| **5** | 🌍 **World Englishes & Commonwealth Nuances** | **SG/MY:** `shiok`, `kiasu`, `lepak`, `makan`, `tapau`, `kopitiam`<br>**UK/AU/CA:** `fortnight`, `chuffed`, `arvo`, `fair_dinkum`, `toque`, `double-double` | Regional lexical anchors and divergence of shared nouns (*"boot"*, *"scheme"*, *"flat"*, *"pavement"*) across Commonwealth varieties vs. US English defaults. |
| **6** | ⚜️ **Francophone Diatopic Nuances (Québec vs. Paris Region vs. Francophonie)** | **Québec:** `char`, `dépanneur`, `chum`, `blonde`, `tiguidou`, `poudrerie`, `magasiner`<br>**Paris/Verlan:** `meuf`, `chelou`, `relou`, `boulot`<br>**West Africa:** `griot`, `teranga`, `maquis` | How a single language (French) requires distinct tokenization sets and contextual embeddings across Québec, regional Parisian registers, and broader Francophone territories. |
| **7** | 🧭 **Iberian, Lusophone & Germanic Regional Continuums** | **PT-PT vs. PT-BR:** `autocarro` / `ônibus`, `pequeno-almoço` / `café_da_manhã`, `telemóvel` / `celular`<br>**Castilian & Catalan:** `sobremesa`, `seny`, `rauxa`<br>**Regional UK & Germanic:** `canny`, `bairn`, `scran`, `ginnel`, `Fingerspitzengefühl` | Dialectal & diatopic parallels across European vs. Brazilian Portuguese (*"rapariga"*), Spanish vs. Catalan, and regional English tonalities (Geordie/Scouse/Yorkshire) with Germanic roots. |
| **8** | 🦃 **Sensory & Culinary Scene Decomposition ("The Turkey Dinner Experiment")** | `roast_turkey`, `mashed_potatoes`, `roasted_pumpkin`, `cranberry_sauce`, `pan_gravy`, `convection_oven`, `warmed_ceramic_plates`, `maillard_crust` | Tokenizing a physical/sensory scene into compositional ontological slots (`[PROTEIN]`, `[STARCH]`, `[ACID_GARNISH]`, `[VESSEL]`, `[HEAT_SOURCE]`) rather than raw character sequences. |

---

## 📂 Repository Structure

```text
expansive_tokenization/
├── Expansive_Tokenization_Workbook.ipynb                      # 🎯 Main Interactive Workbook (7 Guided Labs + Custom Set Builder)
├── GeneratingFirstText_Tokenization_TokenEmbedding.ipynb      # 📚 Foundational & Upgraded Workbook (Phi-3 Generation, BPE & BERT 768-D Shifts)
├── index.html                                                 # 🌐 Interactive Browser Studio (Test & Build Tokenization Sets Live!)
├── JELLOBEAN_TOKENIZATION_DISPATCH_NOTES.md                   # 📝 Conceptual & Research Notes on Context-of-Interest Tokenization
├── src/
│   └── expansive_tokenization/
│       ├── __init__.py
│       ├── presets.py                                         # Curated Context-of-Interest Token Sets & Polysemy Contrast Corpora
│       └── engine.py                                          # Vocabulary Expansion, Fertility/Fragmentation Audit & Embedding Shift Engine
├── requirements.txt                                           # Python dependencies (transformers, torch, scikit-learn, matplotlib, pandas)
└── README.md                                                  # Documentation, First Principles & Quickstart Guide
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

Open **`Expansive_Tokenization_Workbook.ipynb`** (or **`GeneratingFirstText_Tokenization_TokenEmbedding.ipynb`**) in JupyterLab, VS Code, or Google Colab:

```bash
jupyter notebook Expansive_Tokenization_Workbook.ipynb
```

Or open **`index.html`** directly in your browser to use the **Interactive Expansive Tokenization Studio** with zero setup!

---

## 🧪 What You Will Build in the Workbook

1. **Part 1 — Baseline Causal Text Generation (`Phi-3-mini-4k-instruct`):**
   Understand how a pre-trained language model pairs with its default `AutoTokenizer` to turn prompts into generated tokens.
2. **Part 2 — Inside the Tokenizer & "The Turkey Dinner Experiment":**
   Compare how a default subword tokenizer fragments words vs. how humans classify a scene by functional ontology (`roast_turkey`, `mashed_potatoes`, `roasted_pumpkin`, `pan_gravy`, `cranberry_sauce`, `convection_oven`, `warmed_ceramic_plates`).
3. **Part 3 — 768-Dimensional Token Embeddings (`bert-base-uncased`):**
   Inspect how each token is projected into a 768-D vector and why even identical words (`"the"` at the start vs. middle of a sentence) shift coordinates based on sentence relationships.
4. **Part 4 — Subword Fertility & Semantic Fracture Audit Across All 8 Context Packs:**
   Quantify how severely standard tokenizers fracture Greek mythology, anime/LARP vocabulary, Poe Gothic literature, and regional dialects (measuring **Subword Fertility** and **Semantic Fracture Rate**).
5. **Part 5 — Live Tokenizer Expansion & Compositional Semantic Embedding Initialization:**
   Programmatically inject a custom `TokenizationSet` into a Hugging Face `AutoTokenizer`, resize the model's embedding matrix (`model.resize_token_embeddings`), and initialize new token vectors via **Compositional Semantic Averaging** so the model understands them immediately *without retraining from scratch*.
6. **Part 6 — Measuring 768-D Contextual Embedding Shifts Across Personas, Generations & Dialects:**
   Compute cosine similarity and semantic shift distance ($1 - \cos\theta$) when identical words (*"attention"*, *"weights"*, *"cooked"*, *"bet"*, *"boot"*, *"scheme"*, *"char"*, *"dépanneur"*, *"rapariga"*, *"canny"*, *"bleed"*, *"canon"*) move between contexts.
7. **Part 7 — The Custom Tokenization Set Generator:**
   Author your own fandom, anime, LARP, literary, or dialect tokenization set and export a portable `.json` specification.

---

## 🎓 Credits & Academic Acknowledgments

* **Author & Concept Creator:** **Gwendalynn Lim (婉婷) — "1ightray"**
  * **B.Sc. (Honours) in Applied Computing**, **Singapore Institute of Technology (SIT)**
  * **Advanced Machine Learning & AI (SCTP)**, **Nanyang Technological University (NTU)**
  * **Founder & CTO**, **Eve Count** ([gwen@evecount.com](mailto:gwen@evecount.com) · [GitHub: @evecount](https://github.com/evecount) · [LinkedIn](https://www.linkedin.com/in/gwendalynnlim/))
* **Academic & Educational Gratitude:**
  Special thanks to the faculty, coursework, and academic community at the **Singapore Institute of Technology (SIT)** and **Nanyang Technological University (NTU)** for inspiring the foundational explorations into tokenization, 768-dimensional contextual embeddings, and generative text classification that sparked this workbook.
