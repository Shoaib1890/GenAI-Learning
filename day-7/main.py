import os
import json
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

REQUIRED_WEIGHT = 2
PREFERRED_WEIGHT = 1


skill_aliases = {
        "Python": [
            "python"
        ],

        "REST API Development": [
            "rest api",
            "rest apis",
            "express.js",
            "express"
        ],

        "Databases": [
            "postgresql",
            "mysql",
            "mongodb",
            "relational database",
            "relational databases"
        ],

        "Version Control": [
            "git",
            "github"
        ],

        "Authentication": [
            "authentication",
            "auth"
        ],

        "API Integration": [
            "api integration",
            "integrating backend rest apis",
            "third-party api"
        ],

        "Containerization": [
            "docker",
            "containerized",
            "containers"
        ],

        "Cloud": [
            "aws",
            "azure",
            "gcp",
            "google cloud",
            "microsoft azure"
        ],

        "Caching": [
            "redis",
            "caching",
            "cache"
        ],

        "AI/LLM APIs": [
            "openai",
            "groq",
            "openrouter",
            "llm integration",
            "llm api"
        ],

        "GenAI": [
            "generative ai",
            "genai",
            "llm integration",
            "ai-powered"
        ],

        "RAG / Vector Search": [
            "embeddings",
            "vector database",
            "vector databases",
            "rag",
            "retrieval augmented generation"
        ],

        "Full-stack Development": [
            "full-stack",
            "full stack",
            "frontend and backend",
            "frontend and backend development"
        ]
    }


def load_files():
    with open("resume.txt", "r", encoding="utf-8") as file:
        resume = file.read()

    with open("job_description.txt", "r", encoding="utf-8") as file:
        job_description = file.read()

    return resume, job_description


def extract_requirements(job_description):
    skill_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
You are a job description analyzer.

Extract technical skills and capabilities from the job description.

For every extracted requirement, assign exactly one priority:
- required
- preferred

Return ONLY a JSON object.

The JSON object must contain exactly one key:
"requirements"

"requirements" must be an array of objects.

Each object must contain exactly two keys:
"skill" and "priority".

Example:

{
  "requirements": [
    {
      "skill": "Python",
      "priority": "required"
    },
    {
      "skill": "Docker",
      "priority": "preferred"
    }
  ]
}

Rules:
- Do not include markdown.
- Do not include explanations.
- Do not include text before or after the JSON.
- Do not decide whether the candidate has the skill.
"""
            },
            {
                "role": "user",
                "content": job_description
            }
        ],
        response_format={"type": "json_object"}
    )

    skill_result = json.loads(
        skill_response.choices[0].message.content
    )

    return skill_result["requirements"]

def normalize_requirements():

    requirement_groups = {
        "Python": [
            "Python"
        ],

        "REST API Development": [
            "REST API Development",
            "FastAPI"
        ],

        "Databases": [
            "PostgreSQL",
            "relational databases"
        ],

        "Version Control": [
            "Git",
            "GitHub"
        ],

        "Authentication": [
            "Authentication"
        ],

        "API Integration": [
            "API Integration"
        ],

        "Backend System Design": [
            "Backend System Design"
        ],

        "Containerization": [
            "Docker"
        ],

        "Cloud": [
            "AWS",
            "Azure",
            "GCP"
        ],

        "Caching": [
            "Redis"
        ],

        "AI/LLM APIs": [
            "AI/LLM APIs"
        ],

        "GenAI": [
            "GenAI"
        ],

        "RAG / Vector Search": [
            "Embeddings",
            "Vector Databases",
            "RAG"
        ],

        "Full-stack Development": [
            "Full-stack application development"
        ]
    }

    return requirement_groups

def find_exact_matches(resume, requirement_groups):

    

    normalized_resume = resume.lower()

    matched_groups = []
    missing_groups = []

    for group in requirement_groups:

        aliases = skill_aliases.get(
            group,
            []
        )

        matched = any(
            alias in normalized_resume
            for alias in aliases
        )

        if matched:
            matched_groups.append(group)
        else:
            missing_groups.append(group)

    return matched_groups, missing_groups

def prepare_resume_embeddings(resume, embedding_model):
    resume_chunks = [
        line.strip()
        for line in resume.splitlines()
        if len(line.strip()) > 30
    ]

    resume_embeddings = embedding_model.encode(
        resume_chunks
    )

    return resume_chunks, resume_embeddings

def retrieve_evidence(
    resume_chunks,
    resume_embeddings,
    missing_groups,
    requirement_groups,
    embedding_model
):
   

    semantic_candidates = []

    for group in missing_groups:

        skills = requirement_groups[group]

        # -----------------------------
        # Keyword retrieval
        # -----------------------------

        keywords = skill_aliases.get(
            group,
            []
        )

        keyword_evidence = []

        for chunk in resume_chunks:

            chunk_lower = chunk.lower()

            if any(
                keyword in chunk_lower
                for keyword in keywords
            ):
                keyword_evidence.append({
                    "text": chunk,
                    "similarity": 1.0,
                    "source": "keyword"
                })

        # -----------------------------
        # Semantic retrieval
        # -----------------------------

        semantic_query = (
            f"Experience with {group}: "
            + ", ".join(skills)
        )

        group_embedding = embedding_model.encode(
            semantic_query
        )

        similarities = cos_sim(
            group_embedding,
            resume_embeddings
        )[0]

        top_indices = similarities.argsort(
            descending=True
        )[:3]

        semantic_evidence = [
            {
                "text": resume_chunks[index],
                "similarity": round(
                    similarities[index].item(),
                    3
                ),
                "source": "semantic"
            }
            for index in top_indices
        ]

        # -----------------------------
        # Combine evidence
        # -----------------------------

        combined_evidence = (
            keyword_evidence +
            semantic_evidence
        )

        unique_evidence = {}

        for item in combined_evidence:
            unique_evidence[item["text"]] = item

        evidence = list(
            unique_evidence.values()
        )[:5]

        semantic_candidates.append({
            "group": group,
            "skills": skills,
            "evidence": evidence
        })

    return semantic_candidates


def verify_evidence(semantic_candidates):

    verification_prompt = """
You are a strict technical resume evaluator.

Evaluate each requirement group using ONLY the provided resume evidence.

IMPORTANT RULES:
1. Do not assume experience.
2. Do not infer a technology from a related technology.
3. Related concepts are NOT proof of the required technology.
4. For named technologies such as Docker, Redis, AWS, Azure,
   GCP, FastAPI, embeddings, vector databases and RAG,
   require explicit evidence of that technology.
5. For capability requirements such as backend system design,
   require explicit evidence of architecture, system design,
   scalability, service architecture, database design, caching,
   performance architecture, or similar design decisions.
   Building APIs alone is NOT sufficient.
6. REST APIs built with Express.js can satisfy REST API
   development when the requirement allows "FastAPI or similar".
7. Frontend + backend development can satisfy full-stack development.
8. Using an LLM API can support AI/LLM API experience.
9. Relevance filtering, search, or AI features alone do NOT prove
   embeddings, vector databases, or RAG.
10. Never treat semantic similarity as proof.
11. If the evidence is insufficient, return matched=false.
12. Never invent experience.
13. Return ONLY valid JSON.

Use this exact structure:

{
  "results": [
    {
      "group": "group name",
      "matched": true,
      "reason": "short evidence-based explanation"
    }
  ]
}
"""

    verification_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": verification_prompt
            },
            {
                "role": "user",
                "content": json.dumps(semantic_candidates)
            }
        ],
        response_format={"type": "json_object"}
    )

    verification_result = json.loads(
        verification_response.choices[0].message.content
    )

    return verification_result

def calculate_match_score(
    requirements,
    requirement_groups,
    matched_groups,
    verified_matches
):
    final_matched = list(dict.fromkeys(
        matched_groups + verified_matches
    ))

    final_missing = [
        group
        for group in requirement_groups
        if group not in final_matched
    ]


    group_priorities = {}

    for requirement in requirements:
        skill = requirement["skill"]
        priority = requirement["priority"]

        for group, skills in requirement_groups.items():

            if skill in skills:

                if (
                    group not in group_priorities
                    or priority == "required"
                ):
                    group_priorities[group] = priority

                break

    total_weight = 0
    matched_weight = 0

    for group in requirement_groups:

        priority = group_priorities.get(
            group,
            "preferred"
        )

        weight = (
            REQUIRED_WEIGHT
            if priority == "required"
            else PREFERRED_WEIGHT
        )

        total_weight += weight

        if group in final_matched:
            matched_weight += weight

    if total_weight > 0:
        match_score = round(
            (matched_weight / total_weight) * 100,
            2
        )
    else:
        match_score = 0

    return (
        final_matched,
        final_missing,
        match_score,
        matched_weight,
        total_weight
    )


def generate_recommendations(
    match_score,
    final_matched,
    final_missing
):
    recommendation_prompt = f"""
You are a career advisor helping a software developer
prepare for a job.

Based ONLY on the information below, provide concise,
practical recommendations.

Match Score: {match_score}%

Matched Skills:
{", ".join(final_matched)}

Missing Skills:
{", ".join(final_missing)}

Return ONLY valid JSON in this format:

{{
    "strong_matches": [
        "skill"
    ],
    "skill_gaps": [
        "skill"
    ],
    "recommendations": [
        {{
            "skill": "skill name",
            "reason": "why this skill matters for this job",
            "action": "what the candidate should learn or build"
        }}
    ],
    "summary": "short overall assessment"
}}

Rules:
- Use ONLY the matched and missing skills provided.
- Do not invent additional skills.
- Do not change the match score.
- Prioritize the most important missing skills.
- Keep recommendations practical.
"""

    recommendation_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
You are a practical career advisor.
Return only valid JSON.
Do not invent information.
"""
            },
            {
                "role": "user",
                "content": recommendation_prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(
        recommendation_response.choices[0].message.content
    )

resume, job_description = load_files()

print("Resume loaded:", len(resume), "characters")
print("Job description loaded:", len(job_description), "characters")




requirements = extract_requirements(job_description)
# Normalize related requirements into broader capability groups
requirement_groups = normalize_requirements()

print("\n📋 NORMALIZED REQUIREMENTS")

for group, skills in requirement_groups.items():
    print(f"• {group}: {', '.join(skills)}")

matched_groups, missing_groups = find_exact_matches(
    resume,
    requirement_groups
)

print("\n✅ EXACT / GROUP MATCHES")
for group in matched_groups:
    print("•", group)

print("\n❌ GROUPS NOT FOUND")
for group in missing_groups:
    print("•", group)



resume_chunks, resume_embeddings = prepare_resume_embeddings(
    resume,
    embedding_model
)

semantic_candidates = retrieve_evidence(
    resume_chunks,
    resume_embeddings,
    missing_groups,
    requirement_groups,
    embedding_model
)

# --------------------------------------------------
# LLM EVIDENCE VERIFICATION
# --------------------------------------------------

verification_result = verify_evidence(
    semantic_candidates
)

print("\n🤖 EVIDENCE VERIFICATION")

verified_matches = []


results = verification_result.get("results", [])

if not isinstance(results, list):
    print("⚠️ Unexpected verification format:")
    print(verification_result)
    results = []

for result in results:

    if not isinstance(result, dict):
        print("⚠️ Skipping invalid verification result:", result)
        continue

    group = result.get("group")
    matched = result.get("matched", False)
    reason = result.get("reason", "")

    if not group:
        continue

    if matched:
        verified_matches.append(group)
        status = "✅ MATCH"
    else:
        status = "❌ NOT MATCH"

    print(f"\n{group}")
    print(status)
    print("Reason:", reason)



(
    final_matched,
    final_missing,
    match_score,
    matched_weight,
    total_weight
) = calculate_match_score(
    requirements,
    requirement_groups,
    matched_groups,
    verified_matches
)
# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 50)
print("             FINAL RESUME MATCH")
print("=" * 50)

print(f"\nMatch Score: {match_score}%")
print(
    f"Matched Weight: "
    f"{matched_weight}/{total_weight}"
)

print("\n✅ MATCHED")
for group in final_matched:
    print("•", group)

print("\n❌ MISSING")
for group in final_missing:
    print("•", group)




recommendations = generate_recommendations(
    match_score,
    final_matched,
    final_missing
)

# --------------------------------------------------
# RECOMMENDATION REPORT
# --------------------------------------------------

print("\n" + "=" * 50)
print("             JOB MATCH ANALYSIS")
print("=" * 50)

print(f"\n🎯 Match Score: {match_score}%")

print("\n🔥 STRONG MATCHES")
for skill in recommendations["strong_matches"]:
    print("•", skill)

print("\n⚠️ SKILL GAPS")
for skill in recommendations["skill_gaps"]:
    print("•", skill)

print("\n📚 RECOMMENDATIONS")

for item in recommendations["recommendations"]:
    print(f"\n• {item['skill']}")
    print(f"  Why: {item['reason']}")
    print(f"  Action: {item['action']}")

print("\n💡 SUMMARY")
print(recommendations["summary"])

print("\n" + "=" * 50)