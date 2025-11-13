"""Agent factory functions for creating CrewAI agents"""
from crewai import Agent
from typing import List, Optional

from src.tools import GitLabMCPTool, GoogleDriveMCPTool, RAGMilvusTool
from src.llm import GoToCustomLLM
from src.config.constants import AgentRole
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_gitlab_analyzer_agent(llm: GoToCustomLLM, gitlab_tool: GitLabMCPTool) -> Agent:
    """
    Create an agent specialized in fetching GitLab data using tools.

    Args:
        llm: LLM instance (should support tool calling)
        gitlab_tool: GitLab tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.GITLAB_ANALYZER} agent")

    return Agent(
        role=AgentRole.GITLAB_ANALYZER,
        goal='ONLY use the GitLab Project Analyzer tool to fetch data. NEVER make up, assume, or fabricate ANY information.',
        backstory=(
            'You are a strict data fetcher who ONLY reports information from the GitLab Project Analyzer tool. '
            'CRITICAL RULES YOU MUST FOLLOW:\n'
            '1. You MUST call the GitLab Project Analyzer tool for EVERY request - NO EXCEPTIONS\n'
            '2. You MUST ONLY report data that appears in the tool response - NOTHING ELSE\n'
            '3. You are FORBIDDEN from using your training data, memory, or making assumptions\n'
            '4. You are FORBIDDEN from fabricating, inferring, or "filling in" missing information\n'
            '5. If the tool returns an error or no data, you MUST report: "Tool returned no data for this field"\n'
            '6. You MUST wait for the tool response before providing ANY answer\n'
            '7. You MUST include the raw tool output in your response as proof\n\n'
            'VERIFICATION: Before responding, ask yourself:\n'
            '- Did I call the tool? If NO → STOP and call it now\n'
            '- Is this information from the tool response? If NO → DELETE it from your response\n'
            '- Am I making any assumptions? If YES → REMOVE them immediately\n\n'
            'Your ONLY job is to be a transparent conduit for tool data. Nothing more.'
        ),
        tools=[gitlab_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_drive_analyzer_agent(llm: GoToCustomLLM, drive_tool: GoogleDriveMCPTool) -> Agent:
    """
    Create an agent specialized in searching Google Drive for reference documentation.

    Args:
        llm: LLM instance (should support tool calling)
        drive_tool: Google Drive tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.DRIVE_ANALYZER} agent")

    return Agent(
        role=AgentRole.DRIVE_ANALYZER,
        goal='ONLY use the Google Drive Document Analyzer tool to search. NEVER make up, assume, or fabricate ANY documents or content.',
        backstory=(
            'You are a strict document retrieval agent who ONLY reports information from the Google Drive Document Analyzer tool. '
            'CRITICAL RULES YOU MUST FOLLOW:\n'
            '1. You MUST call the Google Drive Document Analyzer tool for EVERY request - NO EXCEPTIONS\n'
            '2. You MUST ONLY report documents that appear in the tool response - NOTHING ELSE\n'
            '3. You are FORBIDDEN from using your training data, memory, or making assumptions about documents\n'
            '4. You are FORBIDDEN from fabricating, inferring, or "filling in" missing document information\n'
            '5. If the tool returns no results, you MUST report: "No documents found in Google Drive"\n'
            '6. You MUST wait for the tool response before providing ANY answer\n'
            '7. You MUST include document URIs (starting with "gdrive:///") as proof\n'
            '8. You are FORBIDDEN from confusing GitLab files with Google Drive documents\n\n'
            'VERIFICATION: Before responding, ask yourself:\n'
            '- Did I call the Google Drive tool (NOT GitLab)? If NO → STOP and call it now\n'
            '- Do the results have URIs starting with "gdrive:///"? If NO → You have wrong tool output\n'
            '- Am I seeing .py files or directories? If YES → You are reporting GitLab data, NOT Drive\n'
            '- Am I making any assumptions about content? If YES → REMOVE them immediately\n\n'
            'Your ONLY job is to be a transparent conduit for Google Drive tool data. Nothing more.'
        ),
        tools=[drive_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_rag_analyzer_agent(llm: GoToCustomLLM, rag_tool: RAGMilvusTool) -> Agent:
    """
    Create an agent specialized in searching internal knowledge base for relevant information.

    Args:
        llm: LLM instance (should support tool calling)
        rag_tool: RAG Milvus tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.RAG_ANALYZER} agent")

    return Agent(
        role=AgentRole.RAG_ANALYZER,
        goal='ONLY use the Internal Knowledge Base Search tool. NEVER use GitLab tool. NEVER make up knowledge base content.',
        backstory=(
            'You are a strict knowledge base retrieval agent who ONLY uses the Internal Knowledge Base Search tool. '
            'CRITICAL - YOU ARE NOT A GITLAB AGENT:\n'
            '- You DO NOT analyze GitLab projects\n'
            '- You DO NOT fetch project files, commits, or README\n'
            '- You DO NOT have access to the GitLab tool\n'
            '- Your ONLY tool is: Internal Knowledge Base Search\n\n'
            'CRITICAL RULES YOU MUST FOLLOW:\n'
            '1. You MUST call the "Internal Knowledge Base Search" tool (NOT GitLab) - NO EXCEPTIONS\n'
            '2. You MUST ONLY report information that appears in the tool response - NOTHING ELSE\n'
            '3. You are FORBIDDEN from using your training data, memory, or making assumptions\n'
            '4. You are FORBIDDEN from fabricating, inferring, or "filling in" missing information\n'
            '5. If the tool returns no results, you MUST report: "No relevant information found in internal knowledge base"\n'
            '6. You MUST wait for the tool response before providing ANY answer\n'
            '7. You MUST verify the response contains "source" fields (like "dge", "genie", "user_income")\n'
            '8. If you see GitLab data (project ID, files, commits), you called the WRONG tool!\n\n'
            'VERIFICATION - BEFORE RESPONDING:\n'
            '- Did I call "Internal Knowledge Base Search" tool? If NO → STOP and call it now\n'
            '- Does the response have a "sources_found" field? If NO → I called the wrong tool!\n'
            '- Am I seeing project_id, files, commits, README? If YES → This is GitLab data, NOT knowledge base!\n'
            '- Did I report which sources were found? If NO → ADD this information\n\n'
            'REMEMBER: You search internal knowledge, NOT GitLab repositories!'
        ),
        tools=[rag_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_documentation_writer_agent(llm: GoToCustomLLM) -> Agent:
    """
    Create an agent specialized in writing documentation based on gathered data.

    Args:
        llm: LLM instance (for content generation)

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.DOCUMENTATION_WRITER} agent")

    return Agent(
        role=AgentRole.DOCUMENTATION_WRITER,
        goal='Generate comprehensive, well-structured documentation in JSON format based on gathered project data',
        backstory=(
            'You are an expert technical writer with deep knowledge of software architecture '
            'and documentation best practices. You take raw data about projects and transform it into '
            'clear, comprehensive documentation that helps developers understand projects quickly. '
            'You synthesize information from multiple sources including GitLab data, Google Drive '
            'reference materials, and internal knowledge base to create thorough documentation. '
            'You always output documentation in valid JSON format with proper structure.'
        ),
        tools=[],  # No tools - only writes documentation based on previous agents' output
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
