from crewai import Agent, Crew, Process, Task, LLM  # Added LLM import
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

@CrewBase
class AiGeminiModel:
    """AiGeminiModel crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, temperature=0.7, model='gemini/gemini-2.0-flash-001'):  # Keep parameters for flexibility
        # Retrieve Gemini API key from environment variable
        google_api_key = os.getenv('GEMINI_API_KEY')
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it before running the application.")

        # Initialize CrewAI's native LLM for Gemini (avoids LangChain prefix bug)
        self.llm = LLM(
            model=model,  # Correct LiteLLM format: 'gemini/gemini-2.0-flash-001'
            api_key=google_api_key,  # Use API key from environment
            temperature=temperature,
        )
        
        # Define agent configurations without LLM (since it's passed in Agent constructors)
        self.agents_config = {
            'researcher': {
                'role': 'Researcher',
                'goal': 'Conduct research on given topics',
                'backstory': 'Expert in data gathering and analysis',
                'verbose': True,
            },
            'reporting_analyst': {
                'role': 'Reporting Analyst',
                'goal': 'Generate reports from data',
                'backstory': 'Skilled in data synthesis and reporting',
                'verbose': True,
            }
        }

        # Define task configurations
        self.tasks_config = {
            'research_task': {
                'description': 'Research the given topic and gather relevant information.',
                'agent': 'researcher',  # Maps to the researcher agent
                'expected_output': 'A detailed research summary.',
            },
            'reporting_task': {
                'description': 'Generate a report based on the research findings.',
                'agent': 'reporting_analyst',  # Maps to the reporting_analyst agent
                'expected_output': 'A formatted report in Markdown.',
            }
        }

    # Agent definitions
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],  # type: ignore[index]
            llm=self.llm,  # Pass CrewAI's native LLM (fixes prefix bug)
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],  # type: ignore[index]
            llm=self.llm,  # Pass CrewAI's native LLM (fixes prefix bug)
            verbose=True
        )

    # Task definitions (unchanged)
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],  # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiGeminiModel crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )