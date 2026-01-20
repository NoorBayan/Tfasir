
# Tafseer Project: Intelligent Heritage Knowledge Engineering

> **From Text Blocks to Knowledge Trees:** Transforming 44 classical Quranic exegesis works into structured, hierarchical, and computable data.

## 📖 Project Overview

The **Tafseer Project** represents the "Cognitive Layer" of the Quranic digital ecosystem. It moves beyond traditional digitization towards **Heritage Knowledge Engineering**. By processing 44 foundational references (such as Al-Tabari, Al-Razi, and Ibn Ashur), the system transforms dense, intertwined textual blocks into **Smart Hierarchical Data Structures**.

This approach allows for the automated decomposition of arguments (Opinion $\to$ Evidence $\to$ Reasoning), rendering this massive heritage "liquid" and computationally processable, while strictly preserving scientific integrity and the map of scholarly divergence.

---

## 💡 Philosophy: Engineering Wisdom

Our Islamic library is filled with treasures of "Mother Tafseers," but they often remain inaccessible behind barriers of complex classical language, length, and overlapping topics.

**Our Mission:** We do not aim to merely "abridge" the text. We aim to **"Engineer Wisdom."** We believe the modern researcher needs the essence of these works without losing their depth. We convert dense texts into **"Liquid Knowledge Units"** that flow easily to the modern mind, answering the "why" and "how" of interpretations without drowning the reader in unstructured details.

---

## ⚠️ The Technical Gap: The Failure of "Flat Summarization"

Most current AI summarization techniques rely on **Flat Summarization**, which compresses text by randomly pruning sentences. This is catastrophic for dialectical heritage texts.

*   **The Problem:** When dealing with a complex argumentative text (e.g., Al-Razi proposing 3 probabilities for a verse), flat summarization often merges them or selects one randomly. This distorts the **Scientific Integrity** and flattens the meaning.
*   **The Solution:** We built a system that understands the **"Tree Logic"** of the exegete (Opinion $\to$ Evidence $\to$ Rebuttal) and reconstructs it computationally into a hierarchical structure (JSON) that preserves the sequence of the argument.

---

## 🚀 Key Contributions & Innovation

We have created what is considered the largest **Structured Summarization Corpus** in the history of digital Quranic service, covering 44 major references.

### 1. Hierarchical Semantic Analysis
We developed algorithms that do not read the text as a "single block" but deconstruct it into a **Knowledge Tree**. The system automatically distinguishes between:
*   Main Idea
*   Sub-idea
*   Transmitted Evidence (Dalil Naqli)
*   Rational Reasoning (Ta'leel Aqli)

### 2. Comparative Intelligence
By converting texts into structured data, we enable **Automated Intersection** between scholars. The system can now answer: *"What are the points of agreement and disagreement between Ibn Ashur and Al-Razi?"* by comparing specific **Knowledge Nodes** rather than abstract text matching.

### 3. Preserving Plurality (Probabilistic Structure)
Unlike reductive summaries, our system is designed to understand the **Structure of Probabilities**. If an interpreter mentions "three faces" of interpretation, the system sorts them as independent branches (Face 1, Face 2, Face 3) with their respective evidence, preserving the rich plurality of the heritage.

---

## 💻 Application & Usage (Interactive Notebook)

We have provided a **Google Colab Notebook** to demonstrate the capabilities of the generated summaries. This notebook allows you to explore the hierarchical summaries, visualize the knowledge trees, and test the filtering mechanisms.

<div align="center">

| **🌟 Explore the Tfasir Summaries** |
| :---: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14800e41Tdxcsb6tcrnn8_wKK-neDcQ86?usp=sharing) |
| *Click the badge above to interact with the data* |

</div>

 <p align="center">
   <img src = "https://raw.githubusercontent.com/NoorBayan/Tfasir/main/images/TfasirColab.png" width = "800px"/>
 </p>

---

## 📂 Data Structure

The dataset is organized to reflect the layered complexity of Quranic exegesis. Each entry represents a structured node containing:

```json
{
  "surah": "Chapter Name",
  "ayah": "Verse Number",
  "interpreter": "Scholar Name",
  "source_book": "Book Title",
  "content_tree": {
    "main_topic": "The core theme of the verse",
    "interpretations": [
      {
        "opinion": "First Interpretation (Face 1)",
        "evidence": "Linguistic or textual evidence",
        "reasoning": "Rational or contextual reasoning"
      },
      {
        "opinion": "Second Interpretation (Face 2)",
        "evidence": "...",
        "reasoning": "..."
      }
    ]
  }
}
```

---

## 🛠 Getting Started

To begin working with the **Tfasir Dataset**:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/NoorBayan/Tfasir.git
    ```
2.  **Run the Notebook**: Access the Google Colab link provided above to visualize the data.
3.  **Explore**: Use the structured JSON files in the `/data` directory for your own NLP or theological research tasks.

---

## 🤝 Contributing

We welcome contributions from NLP engineers, Islamic scholars, and data scientists. Whether it's improving the parsing algorithms, adding new books, or refining the semantic nodes, your help is valuable.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Part of the NoorBayan Initiative for Digital Quranic Intelligence.*

