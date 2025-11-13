# GitLab Documentation Generator with CrewAI

An AI-powered documentation generator that analyzes GitLab projects and creates comprehensive documentation in JSON format using CrewAI agents with dual-LLM architecture.

## Features

- Analyzes GitLab projects using GitLab REST API
- Generates structured documentation in JSON format
- Extracts project metadata, file structure, and recent activity
- Uses CrewAI with dual-LLM architecture (gpt-oss + sahabat-4bit)
- Customizable output format and content
- Supports self-hosted GitLab instances

## Prerequisites

- Python 3.10 or higher
- GitLab Personal Access Token (with `read_api` scope)
- Access to GoTo LiteLLM endpoint (gpt-oss and sahabat-4bit models)

## Installation

1. Clone or navigate to this directory:
```bash
cd /Users/luthfi.reynaldi/Documents/goto-hacks-2025
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Then edit `.env` and add your credentials:
```
GITLAB_TOKEN=your_gitlab_personal_access_token
GITLAB_URL=https://source.golabs.io
```

### Getting a GitLab Personal Access Token

1. Go to your GitLab instance: `https://source.golabs.io/-/user_settings/personal_access_tokens`
2. Click "Add new token"
3. Give it a name and select scopes: `read_api`, `read_repository`
4. Set expiration date as needed
5. Click "Create personal access token"
6. Copy the token and add it to your `.env` file

## Usage

### Basic Usage

Generate documentation for any GitLab project (public or accessible with your token):

```bash
python documentation_agent.py namespace/project
```

Example:
```bash
python documentation_agent.py gopay-ds/Growth/gopay-dge-ride-model_pipeline-staging
```

This will:
1. Analyze the project using GitLab REST API
2. Generate comprehensive documentation using CrewAI with dual-LLM architecture
3. Save the output to `documentation_namespace_project.json`

### Programmatic Usage

You can also use the documentation generator in your own Python code:

```python
from documentation_agent import DocumentationCrew

# Create the documentation crew
doc_crew = DocumentationCrew()

# Generate documentation for a GitLab project
documentation = doc_crew.generate_documentation("namespace/project")

# Save to file
doc_crew.save_documentation(documentation, "output.json")
```

## Output Format

The generated documentation is in JSON format and includes:

```json
{
  "overview": {
    "name": "Project Name",
    "description": "Description",
    "purpose": "Main purpose",
    "default_branch": "main",
    "visibility": "private",
    "license": "MIT"
  },
  "features": ["Feature 1", "Feature 2"],
  "tech_stack": {
    "topics": ["python", "machine-learning"],
    "dependencies": "..."
  },
  "structure": {
    "main_files": [
      {"name": "setup.py", "purpose": "Package setup"},
      {"name": "src/main.py", "purpose": "Main application"}
    ]
  },
  "activity": {
    "stars": 0,
    "forks": 0,
    "open_issues": 0,
    "last_activity": "2024-01-15T10:30:00Z"
  },
  "getting_started": {
    "installation": "...",
    "usage": "...",
    "project_url": "https://source.golabs.io/namespace/project"
  }
}
```

## Project Structure

```
goto-hacks-2025/
├── documentation_agent.py   # Main CrewAI agent implementation
├── github_mcp_tool.py       # GitLab API tool wrapper
├── goto_custom_llm.py       # Custom LLM integration for GoTo models
├── test_gitlab_api.py       # GitLab API validation script
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## How It Works

1. **GitLab API Tool** (`github_mcp_tool.py`):
   - Wraps GitLab REST API access
   - Fetches project info, file structure, commits, and README
   - Provides structured data to the CrewAI agents

2. **Documentation Agent** (`documentation_agent.py`):
   - Two-agent CrewAI system using dual LLM architecture:
     - **Agent 1 (GitLab Data Analyzer)**: Uses `gpt-oss` (GPT OSS 120B) for tool calling and data fetching
     - **Agent 2 (Documentation Writer)**: Uses `sahabat-4bit` (Sahabat AI 70B 4-bit) for documentation writing
   - Analyzes project data and generates comprehensive JSON documentation

3. **Crew Execution**:
   - Sequential process with two specialized agents
   - Agent 1 fetches all project data using GitLab API
   - Agent 2 writes structured documentation based on fetched data
   - Returns formatted JSON documentation

## Customization

### Modify the LLM Models

Edit `documentation_agent.py` to use different models or adjust parameters:

```python
# For data fetching agent (requires tool calling support)
self.gpt_oss_llm = GoToCustomLLM(
    model="openai/gpt-oss-120b",
    endpoint="https://litellm-staging.gopay.sh",
    temperature=1.0  # Adjust as needed
)

# For documentation writing agent
self.sahabat_llm = GoToCustomLLM(
    model="GoToCompany/Llama-Sahabat-AI-v2-70B-IT-awq-4bit",
    endpoint="https://litellm-staging.gopay.sh",
    temperature=0.6  # Lower for more focused output
)
```

### Modify the Agents

Edit `documentation_agent.py` to customize agent behavior:

```python
def create_gitlab_analyzer_agent(self) -> Agent:
    return Agent(
        role='Your Custom Role',
        goal='Your custom goal',
        backstory='Your custom backstory',
        # ... customize as needed
    )
```

### Modify the Output Format

Edit the task description in `create_documentation_task()` to change what information is included in the documentation.

### Add More Tools

Create additional tools in the style of `GitLabMCPTool` and add them to the analyzer agent's `tools` list.

## Troubleshooting

### GitLab API Authentication Errors

If you encounter `403 Forbidden` errors:
- Ensure your `GITLAB_TOKEN` has the correct scopes: `read_api` and `read_repository`
- Verify the token hasn't expired
- Check that you have access to the project you're trying to document

### GitLab API Rate Limiting

If you encounter rate limiting:
- Wait for the rate limit to reset
- Consider adding delays between requests if processing multiple projects
- Check your GitLab instance's rate limit settings

### LLM Endpoint Errors

If you encounter LiteLLM endpoint errors:
- Verify you have access to `https://litellm-staging.gopay.sh`
- Check that the models `openai/gpt-oss-120b` and `GoToCompany/Llama-Sahabat-AI-v2-70B-IT-awq-4bit` are available
- Ensure your network can reach the endpoint

### Project Not Found

If you get "project not found" errors:
- Verify the project path format: `namespace/subgroup/project`
- URL-encode special characters if needed
- Ensure your token has access to the project (check project visibility)

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [GitLab REST API Documentation](https://docs.gitlab.com/ee/api/rest/)
- [GitLab Projects API](https://docs.gitlab.com/ee/api/projects.html)
- [GitLab Repository API](https://docs.gitlab.com/ee/api/repositories.html)
