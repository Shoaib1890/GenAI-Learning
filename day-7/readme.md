# 🤖 AI Job Copilot

An AI-powered resume-to-job matching system that analyzes job requirements, finds evidence from a resume, verifies skill matches, calculates a weighted match score, and generates personalized skill-gap recommendations.

---

## 📌 Overview

AI Job Copilot helps developers understand how well their resume matches a specific job description.

The system combines Large Language Models, structured extraction, semantic embeddings, exact matching, evidence verification, and deterministic Python scoring to produce a practical job-match analysis.

Instead of allowing an LLM to directly decide the final score, the system uses the LLM primarily for natural-language tasks while Python handles the final matching and scoring logic.

This project was built as a hands-on 7-day GenAI learning project, progressively introducing different GenAI concepts and combining them into one practical application.

---

## ✨ Key Features

- Extracts technical requirements from job descriptions
- Classifies requirements as required or preferred
- Normalizes related requirements into skill groups
- Performs exact and alias-based skill matching
- Uses embeddings for semantic evidence retrieval
- Uses an LLM to verify whether resume evidence actually supports a requirement
- Calculates a weighted job-match score
- Identifies candidate skill gaps
- Generates practical learning recommendations
- Uses structured JSON responses from the LLM
- Separates LLM reasoning from deterministic application logic

---

## 🏗️ Architecture

```text
                  Job Description
                         │
                         ▼
              ┌─────────────────────┐
              │ LLM Requirement     │
              │ Extraction          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Requirement         │
              │ Normalization       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Exact / Alias       │
              │ Skill Matching      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Semantic Evidence   │
              │ Retrieval           │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ LLM Evidence        │
              │ Verification        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Deterministic       │
              │ Python Scoring      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Skill Gaps &        │
              │ Recommendations     │
              └─────────────────────┘
```

---

## 🔄 How It Works

1. Load the candidate resume and target job description.
2. Use an LLM to extract technical requirements from the job description.
3. Normalize requirements into predefined skill groups.
4. Perform exact and alias-based matching against the resume.
5. Generate embeddings from resume content.
6. Retrieve potentially relevant resume evidence using semantic similarity.
7. Use an LLM to strictly verify whether the retrieved evidence supports each requirement.
8. Calculate the final weighted match score using Python.
9. Generate practical recommendations for the identified skill gaps.

---

## 🧠 Why Hybrid Matching?

The system combines multiple approaches instead of relying entirely on an LLM or simple keyword matching.

### Exact Matching

Catches skills explicitly mentioned in the resume.

```text
Resume:
Python, PostgreSQL, Git

Requirement:
Python

→ Direct match
```

### Semantic Retrieval

Embeddings help find relevant resume evidence even when the wording differs.

```text
Requirement:
REST API Development

Resume:
Built backend services and integrated REST APIs.

→ Semantically related evidence
```

### LLM Evidence Verification

Semantic similarity is only used to retrieve potential evidence.

The LLM then evaluates whether the evidence actually supports the requirement.

### Deterministic Scoring

Python calculates the final score instead of allowing the LLM to directly generate the score.

This separates probabilistic AI reasoning from deterministic business logic.

---

## 📊 Scoring

Requirements are assigned weights based on their priority:

- Required → weight 2
- Preferred → weight 1

The final score is calculated as:

```text
Match Score =
Matched Requirement Weight
────────────────────────── × 100
Total Requirement Weight
```

Example:

```text
Matched Weight = 13
Total Weight   = 21

Match Score = 13 / 21 × 100
            = 61.9%
```

The final numerical score is calculated by Python, not directly by the LLM.

---

## 🔎 Evidence Verification

The system does not treat semantic similarity as proof that a candidate possesses a skill.

The verification stage follows strict rules:

- Explicit technologies require explicit evidence.
- Related technologies are not automatically treated as equivalent.
- Backend development alone does not automatically prove backend system design.
- AI usage does not automatically prove RAG or vector database experience.
- Insufficient evidence results in a failed match.
- The system does not intentionally invent candidate experience.

This helps reduce false-positive skill matches.

---

## 📚 7-Day GenAI Learning Journey

The project was built progressively over seven days:

- Day 1: LLM API fundamentals with Groq
- Day 2: Structured output and JSON Schema
- Day 3: Embeddings and cosine similarity
- Day 4: RAG fundamentals and retrieval
- Day 5: ChromaDB and vector search
- Day 6: Tool calling and agent loops
- Day 7: Integrated everything into the AI Job Copilot

Each day introduced a new GenAI concept that was applied toward the final system.

---

## 🛠️ Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Core application |
| Groq API | LLM inference |
| GPT-OSS-20B | LLM |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database and retrieval |
| JSON Schema | Structured LLM output |
| python-dotenv | Environment configuration |

---

## 📁 Project Structure

```text
day-7/
│
├── main.py
├── resume.txt
├── job_description.txt
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

`.env` and `.venv` are excluded from Git using `.gitignore`.

For a public repository, use an anonymized/sample resume rather than committing personal resume information.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd day-7
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### 6. Add the input files

Place the following files in the project root:

- `resume.txt`
- `job_description.txt`

### 7. Run the application

```bash
python main.py
```

---

## 📥 Input

The application currently expects two text files.

**resume.txt**  
Contains the candidate's resume in plain-text format.

**job_description.txt**  
Contains the target job description.

The system analyzes both files and produces the final job-match analysis.

---

## 📊 Example Output

```text
==================================================
             FINAL RESUME MATCH
==================================================

Match Score: 61.9%

Matched Weight: 13/21

MATCHED
• Python
• REST API Development
• Databases
• Version Control
• Authentication
• API Integration
• AI/LLM APIs
• GenAI
• Full-stack Development

MISSING
• Backend System Design
• Containerization
• Cloud
• Caching
• RAG / Vector Search
```

The exact score and matched/missing skills depend on the job description and resume being analyzed.

---

## 🎯 Example Recommendations

The system can generate practical recommendations based on the identified skill gaps.

Example:

**Skill Gap:** Docker

**Why:** The job description expects containerized applications.

**Action:** Learn Docker fundamentals and containerize a small FastAPI or Node.js backend application.

Recommendations are generated using the final matched and missing skill information.

---

## 🧪 GenAI Concepts Demonstrated

This project demonstrates practical usage of:

- LLM APIs
- System and user messages
- Structured JSON generation
- JSON Schema
- Prompt engineering
- Embeddings
- Vector representations
- Cosine similarity
- Semantic search
- Retrieval-Augmented Generation (RAG)
- Vector databases
- ChromaDB
- Tool calling
- Agent loops
- LLM-based evidence verification
- Hybrid AI systems
- Deterministic application logic

---

## ⚠️ Limitations

This is currently a learning-focused prototype rather than a production hiring system.

Current limitations include:

- Requirement extraction depends on LLM output.
- Skill normalization currently uses predefined groups and aliases.
- Resume parsing is currently text-based.
- The application currently runs through the command line.
- The scoring system is designed for this prototype.
- Different job descriptions may require additional skill groups or aliases.
- The system should not be used as the sole basis for real hiring decisions.

---

## 🔮 Future Improvements

Possible improvements include:

- Web-based user interface
- Better resume parsing
- More robust requirement normalization
- Persistent vector database
- Resume tailoring for specific jobs
- Job recommendation engine
- Interview question generation
- Job-board integrations
- Historical application tracking
- Resume improvement suggestions
- Automated job matching across multiple job descriptions

---

## 🎓 Key Learnings

This project helped me understand how to combine different GenAI components into a real application rather than treating them as isolated experiments.

Key areas learned:

- How LLM APIs work
- How to enforce structured model output
- How embeddings represent semantic meaning
- How cosine similarity enables semantic comparison
- How RAG retrieves relevant information before generation
- How vector databases support semantic retrieval
- How LLMs can interact with external tools
- How agent loops work
- How to use LLMs for evidence evaluation
- Why deterministic application logic should handle important business rules

---

## 👨‍💻 Author

Mohammed Shoaib  
Software Developer | Full-Stack Development | GenAI

Built as a hands-on 7-day GenAI learning project.