"""Documentation Generation Agent using CrewAI"""
import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from github_mcp_tool import GitHubMCPTool
from goto_custom_llm import GoToCustomLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_provider():
    """
    Get the configured LLM provider based on environment variable.

    Returns:
        LLM instance (either GoToCustomLLM or CrewAI's LLM with Gemini)
    """
    provider = os.getenv("LLM_PROVIDER", "goto").lower()

    if provider == "gemini":
        # Use Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required when LLM_PROVIDER=gemini. "
                "Please set it in your .env file."
            )
        print(f"Using Google Gemini as LLM provider")
        return LLM(
            model="gemini/gemini-2.0-flash-exp",
            api_key=api_key,
            temperature=0.7
        )
    else:  # Default to goto
        # Use GoToCompany custom LLM
        print(f"Using GoToCompany custom LLM as provider")
        return GoToCustomLLM(
            model="GoToCompany/Llama-Sahabat-AI-v2-70B-R",
            endpoint="https://litellm-staging.gopay.sh",
            temperature=0.6
        )


class DocumentationCrew:
    """CrewAI setup for generating documentation from GitHub repositories"""

    def __init__(self):
        self.github_tool = GitHubMCPTool()

        # Get LLM based on LLM_PROVIDER environment variable
        self.llm = get_llm_provider()

    def create_documentation_agent(self) -> Agent:
        """Create an agent specialized in generating documentation"""
        return Agent(
            role='Technical Documentation Specialist',
            goal='Generate comprehensive, well-structured documentation in JSON format based on GitHub repository analysis',
            backstory=(
                'You are an expert technical writer with deep knowledge of software architecture '
                'and documentation best practices. You excel at analyzing codebases and creating '
                'clear, comprehensive documentation that helps developers understand projects quickly. '
                'You always output documentation in valid JSON format with proper structure.'
            ),
            tools=[self.github_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_documentation_task(self, agent: Agent, repository: str) -> Task:
        """Create a task for generating documentation"""
        return Task(
            description=(
                f'TASK: Analyze the GitHub repository "{repository}" and generate comprehensive documentation.\n\n'
                f'STEPS:\n'
                f'1. Use the GitHub Repository Analyzer tool to fetch repository data\n'
                f'2. Analyze the data received from the tool\n'
                f'3. Create a comprehensive JSON document with these sections:\n'
                f'   - overview: {{name, description, purpose, language, license}}\n'
                f'   - features: [list of key features]\n'
                f'   - tech_stack: {{language, topics, dependencies}}\n'
                f'   - structure: {{main_files: [files with descriptions]}}\n'
                f'   - activity: {{stars, forks, open_issues, last_updated}}\n'
                f'   - getting_started: {{installation, usage, repository_url}}\n\n'
                f'IMPORTANT: Return ONLY the JSON object, no markdown code blocks, no extra text. '
                f'Start your response directly with {{.'
            ),
            expected_output=(
                'A valid, complete JSON object (not wrapped in markdown code blocks) following this structure:\n'
                '{{\n'
                '  "overview": {{"name": "...", "description": "...", "purpose": "...", "language": "...", "license": "..."}},\n'
                '  "features": ["feature1", "feature2"],\n'
                '  "tech_stack": {{"language": "...", "topics": [], "dependencies": "..."}},\n'
                '  "structure": {{"main_files": [{{"name": "...", "purpose": "..."}}]}},\n'
                '  "activity": {{"stars": 0, "forks": 0, "open_issues": 0, "last_updated": "..."}},\n'
                '  "getting_started": {{"installation": "...", "usage": "...", "repository_url": "..."}}\n'
                '}}'
            ),
            agent=agent
        )

    def generate_documentation(self, repository: str) -> dict:
        """
        Generate documentation for a given GitHub repository

        Args:
            repository: GitHub repository in format 'owner/repo'

        Returns:
            Dictionary containing the generated documentation
        """
        # Create agent and task
        agent = self.create_documentation_agent()
        task = self.create_documentation_task(agent, repository)

        # Create crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the crew
        result = crew.kickoff()

        # Try to parse the result as JSON
        try:
            # Convert result to string
            result_str = str(result)

            # Strip markdown code blocks if present
            # Handles: ```json\n{...}\n``` or ```\n{...}\n```
            if result_str.strip().startswith('```'):
                # Find the first newline after ```
                start = result_str.find('\n')
                # Find the last ```
                end = result_str.rfind('```')
                if start != -1 and end != -1:
                    result_str = result_str[start+1:end].strip()

            # Parse as JSON
            doc_json = json.loads(result_str)
            return doc_json
        except json.JSONDecodeError as e:
            # If parsing fails, wrap the result in a JSON structure
            print(f"Warning: Could not parse result as JSON: {e}")
            return {
                "repository": repository,
                "documentation": str(result),
                "format": "text",
                "note": "Documentation could not be parsed as JSON, returning as text"
            }

    def save_documentation(self, documentation: dict, output_file: str = "documentation.json"):
        """Save documentation to a JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documentation, f, indent=2, ensure_ascii=False)
        print(f"\nDocumentation saved to {output_file}")


def main():
    """Main function to run the documentation generation"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python documentation_agent.py <owner/repo>")
        print("Example: python documentation_agent.py openai/gpt-3")
        sys.exit(1)

    repository = sys.argv[1]

    print(f"Generating documentation for repository: {repository}")
    print("=" * 60)

    # Create documentation crew
    doc_crew = DocumentationCrew()

    # Generate documentation
    documentation = doc_crew.generate_documentation(repository)

    # Save to file
    output_file = f"documentation_{repository.replace('/', '_')}.json"
    doc_crew.save_documentation(documentation, output_file)

    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
