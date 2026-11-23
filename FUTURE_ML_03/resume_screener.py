"""
Resume Screening & Candidate Ranking System.
Ponytail mode: Minimal code that fully extracts skills, computes semantic fit, and flags skill gaps.
"""
import os
import re
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Common industry skill lexicon for extraction
# shortcut: Regex dictionary matching used over heavy NER models for instant zero-dependency extraction; upgrade to fine-tuned token classifier if custom certifications/rare tools are needed
COMMON_SKILLS = {
    "python", "r", "sql", "java", "c++", "c#", "javascript", "typescript", "html", "css",
    "react", "node.js", "django", "flask", "fastapi", "spring", "docker", "kubernetes",
    "aws", "azure", "gcp", "git", "linux", "machine learning", "deep learning", "nlp",
    "computer vision", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "spark", "hadoop", "tableau", "power bi", "agile", "scrum", "jira", "ci/cd"
}

def clean_text(text: str) -> str:
    """Strip URLs, emails, sentence punctuation, and normalize spaces while keeping technical skill symbols."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[,;!?()[\]{}:\"']", " ", text)
    text = re.sub(r"\.(?=\s|$)", " ", text)
    text = re.sub(r"[^a-z0-9#+\-\./\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_skills(text: str, skill_set=COMMON_SKILLS) -> set:
    """Extract detected skills matching the skill dictionary using space-padded boundaries."""
    cleaned = f" {clean_text(text)} "
    detected = set()
    for skill in skill_set:
        pattern = r"(?<=\s)" + re.escape(skill) + r"(?=\s)"
        if re.search(pattern, cleaned):
            detected.add(skill)
    return detected

def screen_and_rank(resumes: list[dict], job_description: str, required_skills: list[str] = None) -> pd.DataFrame:
    """
    Rank candidate resumes against a job description.
    Returns DataFrame with rank, scores, matched skills, and skill gaps.
    """
    clean_jd = clean_text(job_description)
    target_skills = set(s.lower() for s in required_skills) if required_skills else extract_skills(clean_jd)

    cleaned_resumes = [clean_text(r["resume"]) for r in resumes]
    corpus = [clean_jd] + cleaned_resumes

    # shortcut: TF-IDF cosine similarity balances keyword density with corpus frequency; upgrade to cross-encoder reranker if candidate pool exceeds 10k
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(corpus)
    cosine_scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    results = []
    for i, r in enumerate(resumes):
        cand_skills = extract_skills(r["resume"])
        matched = target_skills.intersection(cand_skills)
        missing = target_skills.difference(cand_skills)

        skill_ratio = len(matched) / len(target_skills) if target_skills else 0.0
        # Hybrid score: 50% semantic content similarity + 50% hard skill match ratio
        composite_score = round(float((0.5 * cosine_scores[i] + 0.5 * skill_ratio) * 100), 2)

        results.append({
            "candidate_id": r.get("id", f"CAND_{i+1:03d}"),
            "name": r.get("name", f"Candidate_{i+1}"),
            "category": r.get("category", "Applicant"),
            "composite_score": composite_score,
            "text_similarity_pct": round(float(cosine_scores[i] * 100), 2),
            "skill_match_pct": round(float(skill_ratio * 100), 2),
            "matched_skills": sorted(list(matched)),
            "missing_skills": sorted(list(missing))
        })

    df_ranked = pd.DataFrame(results).sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    df_ranked.index += 1
    df_ranked.index.name = "rank"
    return df_ranked

def _run_self_check():
    """Unit test self-check asserting core ranking and skill gap logic."""
    sample_jd = "Senior Machine Learning Engineer proficient in Python, Scikit-learn, PyTorch, Docker, and AWS."
    req_skills = ["python", "scikit-learn", "pytorch", "docker", "aws"]
    sample_candidates = [
        {"id": "C_01", "name": "Alex Mercer", "category": "Data Science", "resume": "Experienced ML Engineer with 5 years building PyTorch and Scikit-learn models in Python. Deployed with Docker on AWS."},
        {"id": "C_02", "name": "Jordan Lee", "category": "HR", "resume": "HR Specialist with experience in talent acquisition, recruitment, benefits, employee relations, and onboarding."},
        {"id": "C_03", "name": "Sam Taylor", "category": "Software Engineer", "resume": "Backend developer writing Python and Java microservices deployed with Docker on Linux. Learning AWS."}
    ]
    ranked = screen_and_rank(sample_candidates, sample_jd, req_skills)
    assert ranked.iloc[0]["candidate_id"] == "C_01" and ranked.iloc[0]["skill_match_pct"] == 100.0
    assert ranked.iloc[-1]["candidate_id"] == "C_02"
    assert "pytorch" in ranked.loc[ranked["candidate_id"] == "C_03", "missing_skills"].values[0]

if __name__ == "__main__":
    _run_self_check()
    parser = argparse.ArgumentParser(description="Screen and rank resumes against a target job description.")
    parser.add_argument("--data", default="data/resumes.csv", help="Path to resumes CSV")
    parser.add_argument("--jd", default=None, help="Target job description")
    parser.add_argument("--skills", default="python,sql,machine learning,deep learning,pandas,scikit-learn,aws", help="Comma-separated required skills")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to display")
    parser.add_argument("--out", default="screening_results.csv", help="Output CSV path for rankings")
    args = parser.parse_args()

    required = [s.strip().lower() for s in args.skills.split(",") if s.strip()]
    job_desc = args.jd if args.jd else f"Job Role opening requiring proficiency in {', '.join(required)}."

    base_dir = os.path.dirname(__file__) or "."
    data_path = os.path.join(base_dir, args.data) if not os.path.isabs(args.data) else args.data
    out_path = os.path.join(base_dir, args.out) if not os.path.isabs(args.out) else args.out

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}.")
        import sys
        sys.exit(1)

    df_raw = pd.read_csv(data_path)
    resumes_list = [
        {"id": f"RES_{i+1:03d}", "name": f"Candidate_{i+1}", "category": row["Category"], "resume": row["Resume"]}
        for i, row in df_raw.iterrows()
    ]

    print(f"\nScreening {len(resumes_list)} candidates against role requirements...")
    print(f"Target Skills: {', '.join(required)}\n")

    ranked_results = screen_and_rank(resumes_list, job_desc, required)
    ranked_results.to_csv(out_path)

    print(f"=== TOP {args.top} CANDIDATES ===")
    for rank_idx, row in ranked_results.head(args.top).iterrows():
        print(f"Rank #{rank_idx} | {row['candidate_id']} ({row['category']})")
        print(f"  Fit Score: {row['composite_score']:.1f}% (Text Match: {row['text_similarity_pct']}%, Skill Match: {row['skill_match_pct']}%)")
        print(f"  Matched Skills ({len(row['matched_skills'])}): {', '.join(row['matched_skills'])}")
        print(f"  Skill Gap ({len(row['missing_skills'])} missing): {', '.join(row['missing_skills']) or 'None (100% Match)'}")
        print("-" * 65)

    print(f"\nFull ranked results saved to: {out_path}")
