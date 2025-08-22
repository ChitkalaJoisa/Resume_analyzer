# Resume Analyzer & Job Application System

A Flask-based web application that helps **recruiters** rank resumes against job descriptions and allows **job seekers** to check which required skills are present or missing from their resumes.

# Features

🌐 Web UI with Recruiter & Job Seeker sections.
📑 Recruiter: Upload job description + multiple resumes → get ranked list (similarity score, extracted skills, names & emails).
🙋 Job Seeker: Upload resume + job description → see present & missing skills.
📊 CSV Export: Recruiter can download ranked results.
🔍 Uses spaCy NER for name/email extraction and TF-IDF + cosine similarity for resume-job matching.
🛠️ Predefined required skills for roles: Web Developer, Data Scientist, Software Engineer.

# Tech Stack

Backend: Flask (Python)
Frontend: HTML, CSS (templates in /templates)
Libraries:
spaCy → Named Entity Recognition
PyPDF2 → Resume text extraction
scikit-learn → TF-IDF, cosine similarity
Werkzeug → Secure file handling

# Usage

Navigate to http://127.0.0.1:5000/ in your browser.
Choose Recruiter or Job Seeker.
Recruiter uploads job description + resumes → gets ranked results + CSV download.
Job Seeker uploads resume + job description → gets skill gap analysis.
