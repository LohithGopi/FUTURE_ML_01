# Automated Customer Support Ticket Classifier & Priority Triage

**Future Interns — Machine Learning Task 2**  
*Author:* Lohith Gopi

Production-grade NLP classification and priority dispatching system for inbound customer support operations.

---

## ⚡ What It Does

1. **Text Normalization & Tokenization:** Cleans raw unstructured email/ticket text, removes markup, and extracts unigrams + bigrams.
2. **Category Classification:** Automatically routes tickets into 5 operational queues:
   - `Technical Support`
   - `Billing & Payments`
   - `Customer Service`
   - `Critical Outages`
   - `Internal / HR`
3. **Priority Tagging (`low` / `medium` / `high`):** Predicts baseline urgency via TF-IDF + Linear Support Vector Classification (`LinearSVC`), reinforced with rule-based SLA escalation for downtime/security keywords.
4. **Millisecond Inference:** Sub-5ms inference latency per ticket without heavy GPU or deep learning dependencies.

---

## 📊 Performance Benchmark

Evaluated on 16,338 real customer tickets (80% train / 20% holdout):

| Task | Metric | Score | Baseline |
| :--- | :--- | :--- | :--- |
| **Category Classification** | Accuracy / Weighted F1 | **76.74%** / **0.768** | 20.00% (Random) |
| **Priority Tagging** | Accuracy / Weighted F1 | **65.15%** / **0.652** | 33.33% (Random) |

- **Billing & Payments:** 94.7% precision.
- **Critical Outages:** 83.5% recall with instant SLA escalation override.

---

## 🚀 Quickstart

### 1. Run Pipeline & Train
```bash
py ticket_system.py
```
Outputs model binaries to `models/category_model.pkl` and `models/priority_model.pkl`.

### 2. Live Python Inference
```python
from ticket_system import classify_ticket

ticket = classify_ticket(
    subject="URGENT: Production database locked",
    body="Our API servers cannot read the DB cluster. Users getting 500 errors."
)

print(ticket)
# Output:
# {
#   'category': 'Technical Support',
#   'priority': 'high',
#   'raw_model_priority': 'high',
#   'escalated_by_sla_rule': False
# }
```

### 3. Interactive Notebook
Launch Jupyter to inspect EDA, feature importances, and confusion matrices:
```bash
jupyter notebook notebook.ipynb
```

---

## 🛠 Project Structure

```text
FUTURE_ML_02/
├── data/
│   └── tickets.csv            # 16,338 English support tickets
├── models/
│   ├── category_model.pkl     # Serialized category pipeline
│   ├── priority_model.pkl     # Serialized priority pipeline
│   └── metrics.json           # Evaluation metrics & classification reports
├── notebook.ipynb             # End-to-end EDA, training, and evaluation
├── ticket_system.py           # Core cleaning, training, triage logic & tests
└── README.md                  # System documentation
```
