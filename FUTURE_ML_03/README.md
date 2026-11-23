# Automated Resume Screening & Candidate Ranking System

**Future Interns — Machine Learning Task 3**  
*Author:* Lohith Gopi

A lightweight NLP-powered resume screening and talent intelligence pipeline that automatically scores applicants against specific job descriptions and isolates skill gaps.

---

## ⚡ Key Capabilities

1. **Resume Normalization & Tokenization:** Cleans unstructured text, strips URLs/markup/punctuation while retaining technical symbols (`c++`, `c#`, `node.js`, `scikit-learn`, `ci/cd`).
2. **Skill Extraction & Matching:** Extracts 40+ key technologies and frameworks from resumes and job descriptions using boundary-aware regex matching.
3. **Hybrid Candidate Ranking:** Computes a composite fit score combining:
   $$\text{Composite Score} = 0.5 \times \text{Cosine Similarity} + 0.5 \times \text{Skill Match Ratio}$$
4. **Actionable Skill Gap Identification:** Explicitly returns missing required competencies for each applicant to accelerate recruiter screening decisions.

---

## 📊 Live Benchmark (962 Resumes)

Tested on the standard 962-resume dataset across 25 industry categories (Data Science, Java Developer, DevOps, Web Design, HR, etc.):

- **Latency:** Screens, vectorizes, and ranks all 962 resumes in **< 1.8 seconds**.
- **Accuracy:** Top 10 slots for a Machine Learning Job Description were 100% occupied by Data Science / ML profiles with 0 false positives.

---

## 🚀 Quickstart

### 1. Run Built-in Verification Test
```bash
py resume_screener.py
```

### 2. Python API Usage
```python
from resume_screener import screen_and_rank

job_desc = "Senior Cloud Engineer proficient in Docker, Kubernetes, AWS, and Linux."
required_skills = ["docker", "kubernetes", "aws", "linux"]

candidates = [
    {"id": "01", "name": "Candidate A", "resume": "Experienced in Docker, Kubernetes, AWS, and Linux pipelines."},
    {"id": "02", "name": "Candidate B", "resume": "HR Recruiter handling onboarding, benefits, and interviews."}
]

ranking_df = screen_and_rank(candidates, job_desc, required_skills)
print(ranking_df[["name", "composite_score", "matched_skills", "missing_skills"]])
```

### 3. Interactive Notebook
```bash
jupyter notebook notebook.ipynb
```

---

## 🛠 Project Structure

```text
FUTURE_ML_03/
├── data/
│   └── resumes.csv            # 962 real resumes across 25 categories
├── notebook.ipynb             # End-to-end EDA, ranking, and gap analysis
├── resume_screener.py         # Production cleaning, skill extraction & ranking
└── README.md                  # Documentation
```
