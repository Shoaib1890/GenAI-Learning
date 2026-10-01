import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

print("Paste the job description. Type END on a new line when finished:")

jd_lines = []

while True:
    line = input()

    if line == "END":
        break

    jd_lines.append(line)

job_description = "\n".join(jd_lines)

print("Paste your resume. Type END on a new line when finished:")

resume_lines = []

while True:
    line = input()

    if line == "END":
        break

    resume_lines.append(line)

resume = "\n".join(resume_lines)

schema = {
    "type": "object",
    "properties": {
        "job_title": {
            "type": "string"
        },
        "experience": {
            "type": "string"
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "responsibilities": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "matched_skills": {
        "type": "array",
        "items": {
            "type": "string"
        }
        },
        "missing_skills": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "job_title",
        "experience",
        "skills",
        "responsibilities",
        "matched_skills",
        "missing_skills"
    ],
    "additionalProperties": False
}
def analyze_job(job_description, resume):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
                You are a job and resume analyzer.

                Analyze the job description and resume provided by the user.

                Extract the job title, required experience, technical skills,
                and responsibilities from the job description.

                Compare the required skills from the job description with the skills
                mentioned in the resume.

                "matched_skills" should contain skills required by the job description
                that are also present in the resume.

                "missing_skills" should contain skills required by the job description
                that are not present in the resume.

                The JSON must contain exactly these fields:

                1. "job_title"
                2. "experience"
                3. "skills"
                4. "responsibilities"
                5. "matched_skills"
                6. "missing_skills"

                Do not include any additional fields.
                """
            },
            {
                "role": "user",
                "content": f"""
                JOB DESCRIPTION:
                {job_description}

                RESUME:
                {resume}
                """
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "job_description",
                "schema": schema
            }
        }
    )   
    content = response.choices[0].message.content

    result = json.loads(content)
    return result

def calculate_match_percentage(result):
    total_skills = len(result["skills"])
    matched_skills = len(result["matched_skills"])

    if total_skills > 0:
        return round((matched_skills / total_skills) * 100, 2)

    return 0

def print_candidate_summary(result, match_percentage):
    matched_names = ", ".join(result["matched_skills"])
    missing_names = ", ".join(result["missing_skills"])

    print("\nCandidate Summary:")
    print(
        f"The candidate matches {match_percentage}% "
        f"of the required technical skills."
    )
    print(f"Strong matches: {matched_names}")
    print(f"Skills to improve: {missing_names}")

result = analyze_job(job_description, resume)  

match_percentage = calculate_match_percentage(result)

print("Job Title:", result["job_title"])
print("Experience:", result["experience"])
print("Skills:", result["skills"])
print("Responsibilities:", result["responsibilities"])
print("Matched Skills:", result["matched_skills"])
print("Missing Skills:", result["missing_skills"])
print("Match Percentage:", f"{match_percentage}%")


print_candidate_summary(result, match_percentage)