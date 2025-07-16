# 🤖 AI Job Application Agent

An AI-powered agent that reads your resume and writes personalized cover letters for job listings. Built with LangChain and OpenAI.

---

## 🚀 Features

- 📄 Parse PDF resumes into searchable knowledge
- 🧠 Use vector similarity to match resume content with job descriptions
- ✍️ Auto-generate tailored cover letters using Claude 3, emphasizes work experience over education/internships
- 📄 Error handling - includes validation and error messages
- 💬 Easy prompt interface to generate applications


## 🛠 Tech Stack

- [LangChain](https://www.langchain.com/)
- [Anthorpic Claude-3](https://www.anthropic.com/claude)
- [ChromaDB](https://www.trychroma.com/)
- Python, PyPDF

---

## 📦 Setup

```bash
# Clone the repo
git clone https://github.com/ouverz/job-agent.git
cd job-agent

# Install dependencies
pip install -r requirements.txt

# Add your resume
mkdir data
put resume.pdf in the data folder