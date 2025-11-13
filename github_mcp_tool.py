"""GitLab MCP Tool for CrewAI"""
import os
import json
from typing import Any, Dict, Type
from urllib.parse import quote_plus
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class GitLabMCPToolSchema(BaseModel):
    """Input schema for GitLabMCPTool."""
    project: str = Field(..., description="Project in format 'namespace/project' or project ID")


class GitLabMCPTool(BaseTool):
    name: str = "GitLab Project Analyzer"
    description: str = (
        "Analyzes GitLab projects to extract information about code structure, "
        "files, commits, and merge requests. Useful for generating comprehensive documentation. "
        "Input should be the project namespace and name in format 'namespace/project'."
    )
    args_schema: Type[BaseModel] = GitLabMCPToolSchema

    def _run(self, project: str) -> str:
        """
        Fetch project information using GitLab REST API

        Args:
            project: Project in format 'namespace/project' or project ID

        Returns:
            JSON string with project information
        """
        try:
            # Get project information
            project_info = self._get_project_info(project)

            # Get file structure
            file_structure = self._get_file_structure(project)

            # Get recent commits (limited)
            commits = self._get_recent_commits(project, limit=5)

            # Get README
            readme = self._get_readme(project)

            result = {
                "project": project,
                "info": project_info,
                "file_structure": file_structure,
                "recent_commits": commits,
                "readme": readme
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_project_info(self, project: str) -> Dict[str, Any]:
        """Get basic project information"""
        try:
            import requests

            gitlab_url = os.getenv('GITLAB_URL', 'https://source.golabs.io')
            token = os.getenv('GITLAB_TOKEN', '')

            if not token:
                return {"error": "GITLAB_TOKEN not set in environment"}

            headers = {'PRIVATE-TOKEN': token}

            # URL encode the project path
            project_encoded = quote_plus(project)

            response = requests.get(
                f'{gitlab_url}/api/v4/projects/{project_encoded}',
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "path": data.get("path"),
                    "path_with_namespace": data.get("path_with_namespace"),
                    "description": data.get("description"),
                    "default_branch": data.get("default_branch"),
                    "visibility": data.get("visibility"),
                    "star_count": data.get("star_count"),
                    "forks_count": data.get("forks_count"),
                    "open_issues_count": data.get("open_issues_count"),
                    "topics": data.get("topics", []),
                    "created_at": data.get("created_at"),
                    "last_activity_at": data.get("last_activity_at"),
                    "web_url": data.get("web_url"),
                    "readme_url": data.get("readme_url"),
                    "license": data.get("license", {}).get("name") if data.get("license") else None
                }
            else:
                return {"error": f"Failed to fetch project info: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_file_structure(self, project: str, path: str = "") -> Dict[str, Any]:
        """Get project file structure"""
        try:
            import requests

            gitlab_url = os.getenv('GITLAB_URL', 'https://source.golabs.io')
            token = os.getenv('GITLAB_TOKEN', '')
            headers = {'PRIVATE-TOKEN': token}

            project_encoded = quote_plus(project)

            response = requests.get(
                f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/tree',
                headers=headers,
                params={'path': path, 'per_page': 20}
            )

            if response.status_code == 200:
                data = response.json()
                structure = []
                for item in data[:20]:  # Limit to first 20 items
                    structure.append({
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "mode": item.get("mode")
                    })
                return {"files": structure}
            else:
                return {"error": f"Failed to fetch file structure: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_recent_commits(self, project: str, limit: int = 5) -> list:
        """Get recent commits"""
        try:
            import requests

            gitlab_url = os.getenv('GITLAB_URL', 'https://source.golabs.io')
            token = os.getenv('GITLAB_TOKEN', '')
            headers = {'PRIVATE-TOKEN': token}

            project_encoded = quote_plus(project)

            response = requests.get(
                f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/commits',
                headers=headers,
                params={'per_page': limit}
            )

            if response.status_code == 200:
                data = response.json()
                commits = []
                for commit in data:
                    commits.append({
                        "id": commit.get("id", "")[:7],
                        "short_id": commit.get("short_id", ""),
                        "title": commit.get("title", ""),
                        "message": commit.get("message", ""),
                        "author_name": commit.get("author_name", ""),
                        "authored_date": commit.get("authored_date", ""),
                        "web_url": commit.get("web_url", "")
                    })
                return commits
            else:
                return [{"error": f"Failed to fetch commits: {response.status_code}"}]
        except Exception as e:
            return [{"error": str(e)}]

    def _get_readme(self, project: str) -> str:
        """Get project README"""
        try:
            import requests

            gitlab_url = os.getenv('GITLAB_URL', 'https://source.golabs.io')
            token = os.getenv('GITLAB_TOKEN', '')
            headers = {'PRIVATE-TOKEN': token}

            project_encoded = quote_plus(project)

            # Try common README filenames
            readme_filenames = ['README.md', 'README', 'readme.md', 'Readme.md']

            for filename in readme_filenames:
                file_encoded = quote_plus(filename)
                response = requests.get(
                    f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/files/{file_encoded}/raw',
                    headers=headers,
                    params={'ref': 'main'}  # Try main branch first
                )

                if response.status_code == 200:
                    # Limit README to first 1000 characters
                    return response.text[:1000] + ("..." if len(response.text) > 1000 else "")
                elif response.status_code == 404:
                    # Try master branch
                    response = requests.get(
                        f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/files/{file_encoded}/raw',
                        headers=headers,
                        params={'ref': 'master'}
                    )
                    if response.status_code == 200:
                        return response.text[:1000] + ("..." if len(response.text) > 1000 else "")

            return "README not found or inaccessible"
        except Exception as e:
            return f"Error fetching README: {str(e)}"
