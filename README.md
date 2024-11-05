# Tfasir Dataset - Summarized Quranic Exegesis Corpus

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Structure and Description](#data-structure-and-description)
3. [Methodology and Development Process](#methodology-and-development-process)
4. [Application and Usage](#application-and-usage)
5. [Getting Started](#getting-started)
6. [Contributing](#contributing)
7. [License and Acknowledgments](#license-and-acknowledgments)

---

### Project Overview
The **Tfasir Dataset** project is a transformative resource for Quranic studies and Islamic scholarship, harnessing state-of-the-art technology to make classical Quranic exegesis accessible, concise, and easily navigable.

**Tfasir**, which means interpretation, is essential for understanding the intricate messages of the Quran, addressing questions of meaning, historical context, and application in daily life. However, traditional Tfasir texts, while profound, often present complex linguistic and theological details that can be challenging for readers without a deep background in classical Arabic and Islamic studies.

To bridge this gap, **Tfasir Dataset** offers a meticulously curated collection of summaries from over 90 prominent Tfasir works, designed to provide accessible, reliable interpretations of Quranic verses. This project utilizes AI advancements, particularly the **Gemini language model by Google**, to generate high-quality summaries, making the depth of Quranic knowledge readily available to educators, students, and researchers, while preserving the integrity and richness of the original interpretations.

### Data Structure and Description
The **Tfasir Dataset** is organized in a detailed, hierarchical structure that reflects the layered complexity of Quranic exegesis. Each entry in the dataset includes:
- **Tfasir Book**: The original source text for the interpretation.
- **Interpreter**: Name of the author or scholar responsible for the Tfasir.
- **Surah (Chapter)**: The Quranic chapter in focus.
- **Ayah (Verse)**: The specific verse being interpreted.
- **Interpretive Hierarchy**: For each ayah, the main topics are identified, branching into relevant subtopics that provide additional theological, legal, and contextual explanations. Each primary topic can expand into subtopics, allowing for nuanced exploration of the Tfasir content.

This hierarchical model enables a comprehensive view, allowing users to drill down into specific interpretations or explore broader thematic explanations as desired.

### Methodology and Development Process
The creation of the **Tfasir Dataset** follows a rigorous, multi-phase methodology to ensure both accuracy and usability:
1. **Data Collection**: Sourcing authoritative Tfasir works covering a diverse range of classical and contemporary interpretations.
2. **Data Preprocessing**: Cleaning, organizing, and structuring the raw texts to ensure they are suitable for summarization and database integration.
3. **Model Fine-tuning**: Using Google’s **Gemini language model**, optimized for Arabic language processing, to generate clear, accurate summaries while maintaining the nuanced meanings of each verse.
4. **Data Structuring and Integration**: Formatting the summarized data into a database and hierarchical corpus to ensure logical flow and ease of access.
5. **Evaluation and Validation**: Implementing a rigorous evaluation process with scholars to verify the accuracy, clarity, and authenticity of the summaries, ensuring that they meet the standards of Islamic scholarship.

### Application and Usage
The **Tfasir Dataset** serves as a versatile tool for Quranic education, research, and personal study. Its applications include:
- **Educational Tools**: Providing a reliable resource for Quranic studies programs, which can be integrated into digital curriculums, classroom materials, and online courses.
- **Research Aid**: Supporting scholars and researchers in accessing a wide array of Quranic interpretations across different Tfasir books in a unified format.
- **Community and Personal Study**: Assisting individuals seeking a deeper understanding of Quranic verses without requiring extensive background knowledge in classical Arabic.

To facilitate usage, a **Google Colab notebook** is available, providing a flexible, interactive platform to explore the Tfasir summaries. This tool allows users to navigate the data, filter by specific criteria, and examine interpretations at both high-level and detailed views.

### Getting Started
To begin working with the **Tfasir Dataset**, follow these steps:
 <p align="center">
   <img src = "https://raw.githubusercontent.com/NoorBayan/Tfasir/main/images/TfasirColab.png" width = "800px"/>
 </p>
 
1. **Clone the Repository**:
   Clone the repository to your local machine using the command:
   ```bash
   git clone https://github.com/NoorBayan/Tfasir.git
   
2. **Access the Google Colab Notebook**:  
   Open the provided Google Colab notebook, which is designed to showcase the dataset’s features and facilitate easy exploration.

3. **Explore the Dataset**:  
   Use the notebook's search and filtering functions to delve into specific Tfasir entries, experiment with query parameters, and discover the wide range of interpretations.

### Contributing
We welcome contributions from scholars, developers, and members of the community interested in enhancing the **Tfasir Dataset**. If you would like to contribute, please submit a pull request or contact the project maintainers. Contributions may include:
- Additional data sources
- Improvements to summarization accuracy
- Expanded functionality and tools for dataset navigation

