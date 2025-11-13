"""Streamlit web interface for GitLab Documentation Generator"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.crew import DocumentationCrew, extract_markdown_from_response
from src.utils.logger import setup_logger
from src.utils.validators import validate_gitlab_project

logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="GitLab Documentation Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FC6D26;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6e6e6e;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #FC6D26;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        background-color: #E24329;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📚 GitLab Documentation Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered documentation generation for GitLab projects using multi-agent collaboration</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("### Project Settings")
    project_input = st.text_input(
        "GitLab Project",
        placeholder="namespace/project-name",
        help="Enter the GitLab project in format: namespace/project-name"
    )

    st.markdown("### Integration Options")
    enable_drive = st.checkbox(
        "Enable Google Drive Search",
        value=False,
        help="Search Google Drive for reference documentation"
    )

    enable_rag = st.checkbox(
        "Enable Internal Knowledge Base",
        value=False,
        help="Search internal Milvus knowledge base for relevant information"
    )

    st.markdown("### Output Settings")
    output_file = st.text_input(
        "Output Filename (optional)",
        placeholder="documentation.md",
        help="Leave empty to auto-generate filename"
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses a **dual-LLM architecture**:
    - **GPT OSS 120B**: For tool calling and data fetching
    - **Sahabat AI 70B**: For documentation writing

    **Agents:**
    1. GitLab Data Analyzer
    2. Google Drive Analyzer (optional)
    3. Internal KB Analyzer (optional)
    4. Documentation Writer
    """)

# File viewer section
with st.expander("📂 View Existing Documentation", expanded=False):
    st.markdown("Load and view previously generated documentation files")

    # List existing markdown files
    existing_files = list(Path(".").glob("documentation_*.md"))

    if existing_files:
        selected_file = st.selectbox(
            "Select a file to view:",
            options=existing_files,
            format_func=lambda x: x.name
        )

        if selected_file and st.button("Load File"):
            with open(selected_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract and clean markdown (in case it has JSON wrapping)
            cleaned_content = extract_markdown_from_response(content)

            st.markdown("---")
            st.markdown("### 📄 File Content")

            # Show in tabs
            view_tab1, view_tab2 = st.tabs(["📖 Preview", "💾 Download"])

            with view_tab1:
                st.markdown(cleaned_content, unsafe_allow_html=False)

            with view_tab2:
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=cleaned_content,
                    file_name=selected_file.name,
                    mime="text/markdown"
                )
    else:
        st.info("No documentation files found. Generate some documentation first!")

st.markdown("---")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🚀 Generate Documentation")

    # Validation feedback
    if project_input:
        if validate_gitlab_project(project_input):
            st.markdown(f'<div class="success-box">✅ Valid project format: <strong>{project_input}</strong></div>', unsafe_allow_html=True)
        else:
            st.error("❌ Invalid project format. Expected: namespace/project-name")

    # Generate button
    generate_button = st.button("🎯 Generate Documentation", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Agent Status")
    agent_count = 2  # Minimum: GitLab + Writer
    if enable_drive:
        agent_count += 1
    if enable_rag:
        agent_count += 1

    st.metric("Active Agents", agent_count)
    st.info(f"""
    **Enabled Agents:**
    - ✅ GitLab Data Analyzer
    - {'✅' if enable_drive else '❌'} Google Drive Analyzer
    - {'✅' if enable_rag else '❌'} Internal KB Analyzer
    - ✅ Documentation Writer
    """)

# Documentation generation
if generate_button:
    if not project_input:
        st.error("⚠️ Please enter a GitLab project path")
    elif not validate_gitlab_project(project_input):
        st.error("⚠️ Invalid project format. Expected: namespace/project-name")
    else:
        try:
            # Progress container
            with st.container():
                st.markdown("---")
                st.markdown("### 🔄 Generation Progress")

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Initialize crew
                status_text.text("Initializing documentation crew...")
                progress_bar.progress(10)

                doc_crew = DocumentationCrew(
                    enable_google_drive=enable_drive,
                    enable_rag=enable_rag
                )

                # Generate documentation
                status_text.text(f"Generating documentation for {project_input}...")
                progress_bar.progress(30)

                with st.spinner(f"Agents are collaborating to generate documentation..."):
                    documentation = doc_crew.generate_documentation(project_input)

                progress_bar.progress(80)
                status_text.text("Saving documentation...")

                # Save documentation
                output_path = doc_crew.save_documentation(
                    documentation,
                    output_file if output_file else None
                )

                progress_bar.progress(100)
                status_text.text("✅ Documentation generated successfully!")

                # Success message
                st.markdown(f'<div class="success-box">✅ <strong>Documentation saved to:</strong> {output_path}</div>', unsafe_allow_html=True)

                # Display documentation
                st.markdown("---")
                st.markdown("### 📄 Generated Documentation")

                # Tabs for different views
                tab1, tab2 = st.tabs(["📖 Preview", "💾 Download"])

                with tab1:
                    # Render markdown properly
                    markdown_content = documentation.get("documentation", "")
                    st.markdown(markdown_content, unsafe_allow_html=False)

                with tab2:
                    st.download_button(
                        label="⬇️ Download Markdown",
                        data=documentation.get("documentation", ""),
                        file_name=os.path.basename(output_path),
                        mime="text/markdown"
                    )

                    st.info(f"File also saved locally at: `{output_path}`")

        except ValueError as e:
            st.error(f"❌ Validation Error: {str(e)}")
            logger.error(f"Validation error: {e}")

        except Exception as e:
            st.error(f"❌ Error generating documentation: {str(e)}")
            logger.error(f"Error generating documentation: {e}", exc_info=True)

            with st.expander("🔍 Error Details"):
                st.code(str(e))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6e6e6e; padding: 1rem;">
    Made with ❤️ using CrewAI, GoTo Custom LLMs, and Streamlit
</div>
""", unsafe_allow_html=True)
