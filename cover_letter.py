import os
from typing import List, Dict, Any
from dataclasses import dataclass

from langchain.agents import initialize_agent, Tool
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv


@dataclass
class AgentConfig:
    """Configuration settings for the AI agent"""
    model_name: str = "claude-3-haiku-20240307"
    temperature: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_search_k: int = 2
    agent_type: str = "zero-shot-react-description"
    verbose: bool = True


class ResumeProcessor:
    """Handles resume loading and vector database operations"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.embedding = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.resume_db = None
    
    def load_resume(self, resume_path: str) -> None:
        """Load resume from PDF and create vector database"""
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
        
        loader = PyPDFLoader(resume_path)
        pages = loader.load_and_split()
        self.resume_db = Chroma.from_documents(pages, self.embedding)
    
    def search_similar_content(self, query: str) -> str:
        """Search for similar content in resume based on query"""
        if not self.resume_db:
            raise ValueError("Resume database not initialized. Call load_resume() first.")
        
        similar_docs = self.resume_db.similarity_search(query, k=self.config.similarity_search_k)
        return " ".join([doc.page_content for doc in similar_docs])


class CoverLetterGenerator:
    """Generates cover letters using AI"""
    
    def __init__(self, llm: ChatAnthropic, resume_processor: ResumeProcessor):
        self.llm = llm
        self.resume_processor = resume_processor
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for cover letter generation"""
        return PromptTemplate(
            input_variables=["job_description", "resume_summary"],
            template="""
You are a job-seeking assistant and a professional cover letter writer. Write a concise, personalized cover letter using the job description and this resume summary. Highlight
key projects, tasks, achievements and milestones from the resume summary if that is applicable and relevant for the job description. I would
avoid mentions of internships, education as the professional experience trumps those. 

Job Description:
{job_description}

Resume Summary:
{resume_summary}
"""
        )
    
    def generate(self, job_description: str) -> Dict[str, Any]:
        """Generate a cover letter for the given job description"""
        resume_summary = self.resume_processor.search_similar_content(job_description)
        
        chain = self.prompt_template | self.llm
        return chain.invoke({
            "job_description": job_description,
            "resume_summary": resume_summary
        })


class CoverLetterAgent:
    """Main AI agent for cover letter generation"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self._setup_environment()
        self.llm = self._initialize_llm()
        self.resume_processor = ResumeProcessor(self.config)
        self.cover_letter_generator = CoverLetterGenerator(self.llm, self.resume_processor)
        self.agent = None
    
    def _setup_environment(self) -> None:
        """Load environment variables"""
        load_dotenv(find_dotenv())
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    
    def _initialize_llm(self) -> ChatAnthropic:
        """Initialize the Claude LLM"""
        return ChatAnthropic(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        return [
            Tool(
                name="CoverLetterTool",
                func=self.cover_letter_generator.generate,
                description="Generates a cover letter from the job description and the resume summary"
            )
        ]
    
    def setup(self, resume_path: str) -> None:
        """Setup the agent with resume data"""
        self.resume_processor.load_resume(resume_path)
        tools = self._create_tools()
        
        self.agent = initialize_agent(
            tools,
            self.llm,
            agent_type=self.config.agent_type,
            verbose=self.config.verbose
        )
    
    def generate_cover_letter(self, job_description: str) -> Dict[str, Any]:
        """Generate a cover letter for the given job description"""
        if not self.agent:
            raise ValueError("Agent not initialized. Call setup() first.")
        
        return self.agent.invoke(f"Write a cover letter for this job: {job_description}")


def main():
    """Example usage of the CoverLetterAgent"""
    
    # Configuration
    config = AgentConfig(
        model_name="claude-3-haiku-20240307",
        temperature=0.3,
        similarity_search_k=2,
        verbose=True
    )
    
    # Initialize agent
    agent = CoverLetterAgent(config)
    agent.setup("data/resume.pdf")
    
    # Example job posting
    job_post = """
Title: Senior Data Analyst 
Basic Requirements:
• Bachelor's degree in Computer Science, Information Systems, Business Administration, or a related field.
• 3-5 years of proven experience working with Power BI in a business intelligence or data analytics role.
• Certification in Microsoft Power BI Data Analyst or similar
• Strong experience with DAX (Data Analysis Expressions) and SQL
• Strong understanding of Project Portfolio Management (PPM) principles and practices.
• Experience with data modeling and integrating data from various systems.
• Experience in managing requirements backlogs and working with Confluence/Jira and collaborate with Scrum Teams
• Excellent analytical and problem-solving skills with a keen attention to detail.
• Ability to communicate effectively with both technical and non-technical stakeholders.
• Strong organizational skills and experience prioritizing competing requirements and managing stakeholder expectations.
• Driver mentality to help get things done
• Advanced English Expression and the
• ability to convey ideas clearly and effectively.
"""
    
    # Generate cover letter
    try:
        result = agent.generate_cover_letter(job_post)
        print("Generated Cover Letter:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"Error generating cover letter: {e}")


if __name__ == "__main__":
    main()