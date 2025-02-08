import re
import yaml
import os
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import litellm


load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY", "your_groq_api_key")  # or replace with your key
os.environ["GROQ_API_KEY"] = groq_api_key


os.environ["OPENAI_API_KEY"] = groq_api_key
GROQ_API_BASE = "https://api.groq.com/openai/v1"  
os.environ["OPENAI_API_BASE"] = GROQ_API_BASE
os.environ["OPENAI_MODEL_NAME"] = "groq/gemma2-9b-it"


litellm.api_key = groq_api_key
litellm.api_base = GROQ_API_BASE

litellm.model_alias_map = {
    "gpt-40-mini": "groq/gemma2-9b-it"
}

print("Using Groq API Key:", os.environ["GROQ_API_KEY"])
print("Using API Base:", litellm.api_base)
print("Model mapping:", litellm.model_alias_map)


current_dir = Path.cwd()
agents_config_path = current_dir / "config" / "agents.yaml"
tasks_config_path = current_dir / "config" / "tasks.yaml"


with open(agents_config_path, "r") as file:
    agents_config = yaml.safe_load(file)

with open(tasks_config_path, "r") as file:
    tasks_config = yaml.safe_load(file)


def create_agent(agent_name):
    return Agent(
        role=agents_config[agent_name]["role"],
        goal=agents_config[agent_name]["goal"],
        backstory=agents_config[agent_name]["backstory"],
        allow_delegation=False,
        verbose=True,
    )


spamfilter = create_agent("spamfilter")
analyst = create_agent("analyst")
scriptwriter = create_agent("scriptwriter")
formatter = create_agent("formatter")
scorer = create_agent("scorer")


discussion =  """
From: john.doe@company.com (John Doe)
Subject: Team Meeting - Project Roadmap Discussion

			PROJECT ROADMAP DISCUSSION

		Thursday, March 14, 2025

	   Conference Room A, XYZ Corporation

SPONSOR: The XYZ Corporation Product Development team is organizing a 
strategy meeting to discuss the next phase of our project roadmap. 
The goal of the meeting is to align on key milestones, review any 
challenges, and finalize action items for the upcoming quarter.

AGENDA:
- Review of the current project status
- Discussion of upcoming deliverables and deadlines
- Identification of potential risks and mitigation strategies
- Open Q&A session for team input

ATTENDEES: This meeting is intended for all team members actively involved 
in the project, including engineers, designers, and product managers.

LOCATION: The meeting will take place in Conference Room A at our headquarters.
A virtual meeting link will be provided for remote participants.

RSVP: Please confirm your availability by March 10, 2025 by replying to this email 
or contacting john.doe@company.com.

For any questions, feel free to reach out. Looking forward to a productive discussion!

		Best regards,
		John Doe
		Product Manager
		XYZ Corporation
"""


task0 = Task(
    description=tasks_config["task0"]["description"].replace("{{discussion}}", discussion),
    expected_output=tasks_config["task0"]["expected_output"],
    agent=spamfilter,
)

crew0 = Crew(agents=[spamfilter], tasks=[task0], process=Process.sequential, verbose=True)
result = crew0.kickoff()


spam_output = str(result.tasks_output).strip().upper()
print("Spamfilter Output:", spam_output)


if spam_output in ["SPAM", "VULGAR"]:
    print("This message is classified as spam and will be filtered out.")
else:
    print("Not Spam. Proceeding with further tasks...")
    
  
    task1 = Task(
        description=tasks_config["task1"]["description"].replace("{{discussion}}", discussion),
        expected_output=tasks_config["task1"]["expected_output"],
        agent=analyst,
    )

  
    task2 = Task(
        description=tasks_config["task2"]["description"].replace("{{discussion}}", discussion),
        expected_output=tasks_config["task2"]["expected_output"],
        agent=scriptwriter,
    )

 
    task3 = Task(
        description=tasks_config["task3"]["description"],
        expected_output=tasks_config["task3"]["expected_output"],
        agent=formatter,
    )

    crew = Crew(
        agents=[analyst, scriptwriter, formatter],
        tasks=[task1, task2, task3],
        verbose=True,
        process=Process.sequential,
    )

    result = crew.kickoff()
    result_text = str(result.tasks_output)  

    task4 = Task(
    description=tasks_config["task4"]["description"].replace("{{script}}", result_text),
    expected_output=tasks_config["task4"]["expected_output"],
    agent=scorer,
    )

    crew4 = Crew(agents=[scorer], tasks=[task4], process=Process.sequential, verbose=True)
    score = crew4.kickoff()

    print("Final Script:\n", result_text)
    print("Script Score:", score)



    
    score_output = str(score).strip()

    print("Final Script:\n", result_text)
    print("Script Evaluation:", score_output)
