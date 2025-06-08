<div align="center">
  <img src="https://github.com/user-attachments/assets/24239b8c-5ae6-4c7b-adf4-5da7c0f2469d" width="450" alt="agent">
  <br>
  <h3>LLMFlow: Build Goal-Driven AI Agents with Function Calling</h3>
</div>


LLMFlow is a flexible framework for creating AI agents that use LLM and function calls to achieve goals. Based on the GAME methodology (Goals, Actions, Memory, Environment) with an iterative Loop, it enables agents to work with various tools and services. A tag system helps organize and connect tools dynamically.

<p align="center">
  <img src="https://github.com/user-attachments/assets/949d6fdf-24f1-4d5e-9165-3113edddc432" alt="layout" width="500" />
</p>

The agent sets Goals (what to accomplish) and prioritizes them. To achieve these, it plans and executes Actions (computations, service requests, or other operations). All outcomes—successes, failures, relevant context—are stored in Memory so it can learn from past steps and avoid repeating mistakes. An Environment layer abstracts interactions with outside systems (APIs, databases, files) behind a stable interface. These components operate in a continuous Loop: review goals, consult memory, perform the next action via the environment, record the results, and reevaluate goals—repeating until the objectives are met.

The Agent is responsible for managing the overall flow of interactions, coordinating goals, actions, and memory, as well as selecting and executing the necessary tools; goals are created and updated dynamically, managed by priority, and their completion is tracked; memory provides conversation history storage, state persistence, and context management for tool operations; the environment handles tool registration and discovery, resource management, and integration with external services.

The architecture follows the state machine (or more generally, process automaton) principle, where the system is always aware of its current state and the possible transitions. Each layer — Input, Goals, Memory, Actions, Environment, Feedback, Output — represents a logical processing stage. Transitions occur both within layers (e.g. Goal Planning → Subgoal Planning) and between layers (e.g. Feedback influencing Memory to trigger a new cycle).
  
### Capabilities
  - OpenAI API support (GPT-4o, GPT-4o-mini)
  - Generic provider support for custom LLMs
  - Automatic tool registration and discovery
  - Error handling and recovery
  - Context-aware responses
  - Goal tracking and completion

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/KazKozDev/LLMFlow.git
cd LLMFlow

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

#### LLM Setup
1. Copy the example config:
   ```bash
   cp llm_config.yaml.example llm_config.yaml
   ```
2. Configure your LLM:
   ```yaml
   provider_config:
     provider: "openai"
     api_key: "your-api-key"  # Or use OPENAI_API_KEY env var
     model: "gpt-4o-mini"
   ```

#### Environment Variables
1. Copy the example env file:
   ```bash
   cp llmflow/env.example .env
   ```
2. Configure required variables based on tools you'll use:
   ```env
   # OpenAI
   OPENAI_API_KEY=your-key

   # Google Cloud (for various tools)
   GOOGLE_CREDENTIALS_FILE=credentials.json
   GOOGLE_TOKEN_FILE=token.json

   # Email Configuration
   EMAIL_SMTP_SERVER=smtp.gmail.com
   EMAIL_USERNAME=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password

   # Additional services as needed...
   ```

### 3. Running

#### Interactive Mode
```bash
python main_example.py [--model MODEL_NAME] [--max-iterations N]
```
> Note: `main_example.py` is a demonstration file that shows how to use the LLMFlow framework. It provides a basic interactive chat interface with the AI agent. You can use it as a reference for building your own applications.

#### Simulation Mode
```bash
python run_simulation.py
```

### Tag System
The framework uses a flexible tagging system to categorize and organize tools. Tags help the AI agent understand tool capabilities and select the most appropriate tool for each task. Each tool can have multiple tags, including category tags (e.g., "system", "web", "data"), capability tags (e.g., "read", "write", "analyze"), and domain-specific tags (e.g., "email", "database", "api"). The agent uses these tags for intelligent tool selection and chaining.

### Tool Categories
- **System**: File operations, system monitoring, shell commands
- **Web**: Search, content parsing, API interactions
- **Data**: Analysis, database operations, embeddings
- **Communication**: Email, messaging, calendars
- **Media**: Speech recognition, text-to-speech, image processing
- **Development**: Code execution, debugging, version control
- **Cloud**: AWS, Google Cloud, Azure integrations

<div align="center">
  <img src="https://github.com/user-attachments/assets/0aa88988-9b27-47e0-8d01-0b4075eb4dcd" width="450" alt="agent">
</div>

### Available Tools

LLMFlow provides a comprehensive set of tools (116 in total) organized by category. Below is the complete list of available tools:


<details>
<summary>File System & IO Operations</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| read_file | Read content from a file | file_system, read |
| write_file | Write content to a file | file_system, write |
| list_directory | List contents of a directory | file_system, list |
| read_file_advanced | Read file with format detection and security validation | file_system, read, file_io |
| write_file_advanced | Write to file with format detection and atomic writes | write, file_system, file_io |
| convert_file_format | Convert between file formats (CSV, JSON, XML, YAML) | file_system, convert, file_io |
| get_file_info | Get detailed file information | file_system, info, file_io |

</details>

<details>
<summary>System Operations & Monitoring</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| get_current_system_metrics | Get current system resource metrics | system, monitoring, metrics |
| get_detailed_resource_usage | Get detailed system resource usage | system, monitoring, resources |
| get_top_processes | Get top resource-consuming processes | system, monitoring, processes |
| start_system_monitoring | Start background system monitoring | system, monitoring, background |
| stop_system_monitoring | Stop background system monitoring | system, monitoring, background |
| get_monitoring_summary | Get summary of system monitoring data | system, monitoring, summary |
| add_system_alert | Add a system alert condition | system, monitoring, alerts |
| list_system_alerts | List configured system alerts | system, monitoring, alerts |
| get_comprehensive_system_info | Get comprehensive system information | system, info, detailed |
| get_monitoring_statistics | Get statistical analysis of monitoring data | system, monitoring, statistics |
| system_health_check | Run system health check | system, monitoring, health |
| analyze_system_performance | Analyze system performance | system, monitoring, performance |

</details>

<details>
<summary>Shell & Command Execution</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| execute_shell_command | Execute a shell command | shell, command, execute |
| execute_background_command | Execute a command in background | shell, background, command |
| get_process_status | Get status of a process | shell, process, status |
| kill_process | Kill a running process | shell, process, kill |
| list_running_processes | List all running processes | shell, process, list |
| execute_script | Execute a script file | shell, script, execute |
| get_system_info | Get basic system information | shell, system, info |
| get_shell_execution_stats | Get shell command execution statistics | shell, stats, history |
| change_working_directory | Change current working directory | shell, directory, navigation |
| set_environment_variable | Set an environment variable | shell, environment, config |
| get_environment_variables | Get environment variables | shell, environment, variables |
| quick_shell_command | Execute a simple shell command | shell, quick, command |
| configure_shell_security | Configure shell security settings | shell, security, config |

</details>

<details>
<summary>Web & Search</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| search_duckduckgo | Search the web with DuckDuckGo | search, web, duckduckgo |
| quick_search | Quick web search with simplified results | search, web, quick |
| parse_selected_urls | Parse and extract content from URLs | web, parse, extract |

</details>

<details>
<summary>Cloud Services</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| quick_setup | Quick setup for Google Cloud services | setup, cloud, google |
| batch_upload_folder | Upload folder contents to Google Drive | upload, batch, cloud, google |
| create_document_with_content | Create Google Doc with content | document, create, cloud, google |
| create_data_spreadsheet | Create Google Sheet with data | spreadsheet, create, cloud, google |
| upload_file_to_drive | Upload file to Google Drive | upload, file, cloud, google |
| download_file_from_drive | Download file from Google Drive | download, file, cloud, google |
| create_google_document | Create Google Doc | document, create, cloud, google |
| create_google_spreadsheet | Create Google Sheet | spreadsheet, create, cloud, google |
| search_google_drive | Search files in Google Drive | search, cloud, google |
| list_google_drive_files | List files in Google Drive | list, cloud, google |
| create_google_drive_folder | Create folder in Google Drive | folder, create, cloud, google |
| share_google_drive_file | Share Google Drive file | share, cloud, google |
| edit_google_document | Edit Google Doc | document, edit, cloud, google |
| read_google_document | Read Google Doc content | document, read, cloud, google |
| read_google_spreadsheet | Read Google Sheet data | spreadsheet, read, cloud, google |
| write_google_spreadsheet | Write data to Google Sheet | spreadsheet, write, cloud, google |
| backup_google_drive | Backup Google Drive files | backup, cloud, google |
| sync_with_google_drive | Sync local files with Google Drive | sync, cloud, google |
| get_google_drive_storage | Get Google Drive storage info | storage, info, cloud, google |
| test_google_cloud_connection | Test Google Cloud connection | test, connection, cloud, google |
| create_google_cloud_agent | Create Google Cloud agent | agent, cloud, google |

</details>

<details>
<summary>Messaging & Communication</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| quick_slack_setup | Quick Slack integration setup | setup, slack, messaging |
| quick_discord_setup | Quick Discord integration setup | setup, discord, messaging |
| quick_telegram_setup | Quick Telegram bot setup | setup, telegram, messaging |
| setup_multi_platform_agent | Setup messaging agent for multiple platforms | setup, messaging, multi-platform |
| send_slack_message | Send message to Slack | send, slack, messaging |
| send_discord_message | Send message to Discord | send, discord, messaging |
| send_telegram_message | Send message via Telegram bot | send, telegram, messaging |
| broadcast_to_all_platforms | Broadcast message to all platforms | broadcast, messaging, multi-platform |
| upload_file_to_slack | Upload file to Slack | upload, slack, file |
| upload_file_to_discord | Upload file to Discord | upload, discord, file |
| upload_file_to_telegram | Upload file via Telegram bot | upload, telegram, file |
| get_slack_channels | Get available Slack channels | channels, slack, list |
| get_discord_channels | Get available Discord channels | channels, discord, list |
| get_slack_messages | Get messages from Slack channel | messages, slack, read |
| get_discord_messages | Get messages from Discord channel | messages, discord, read |
| get_telegram_updates | Get updates from Telegram bot | updates, telegram, read |
| test_platform_connections | Test messaging platform connections | test, messaging, connections |
| create_messenger_agent | Create messaging agent | agent, messaging, wrapper |

</details>

<details>
<summary>Email</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| quick_send_email | Quick email sending function | communication, email, quick, send |
| quick_read_emails | Quick email reading function | communication, email, quick, read |
| setup_gmail_config | Setup Gmail configuration | setup, email, gmail, configuration |
| setup_outlook_config | Setup Outlook/Hotmail configuration | setup, email, outlook, configuration |
| send_email_advanced | Send advanced email with full features | communication, email, advanced, send |
| read_emails_advanced | Read emails with advanced filtering | communication, email, advanced, read |
| send_bulk_emails | Send bulk emails with personalization | communication, email, bulk, marketing |
| search_emails | Search emails with specific criteria | search, email, advanced, filter |
| save_email_attachment | Save an email attachment to file | save, email, attachments, download |
| save_email_template | Save an email template for reuse | save, email, automation, template |
| list_email_folders | List all available email folders | management, email, list, folders |
| export_emails_to_file | Export emails to a file | export, data, email, backup |
| get_email_stats | Get email statistics and account info | stats, info, email, analytics |
| create_email_agent | Create an email agent for complex operations | agent, email, wrapper, advanced |

</details>

<details>
<summary>Vector Embeddings & Search</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| quick_embed | Quick embedding generation | text, embeddings, vectorization, ai |
| quick_search_setup | Quick setup of vector search | vector, embeddings, setup, search |
| similarity_search | Quick similarity search | text, embeddings, similarity, search |
| add_texts_to_vector_store | Add texts to a vector store | text, embeddings, add, vector_store |
| search_similar_texts | Search for similar texts | vector_store, embeddings, query, search |
| cluster_text_embeddings | Cluster texts based on embeddings | clustering, embeddings, ml, analysis |
| reduce_embedding_dimensions | Reduce dimensionality of embeddings | embeddings, dimensionality_reduction, analysis |
| export_vector_data | Export vector store data to file | export, embeddings, persistence, data |
| import_vector_data | Import vector store data from file | persistence, embeddings, import, data |
| get_embedding_stats | Get statistics about embeddings | stats, embeddings, info, analysis |
| create_embeddings_agent | Create embeddings and vector search agent | agent, embeddings, wrapper, advanced |

</details>

<details>
<summary>Media Processing</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| download_youtube_video | Download video from YouTube | download, youtube, media |
| transcribe_audio | Transcribe audio to text | speech-to-text, transcription, audio |
| process_video | Process video with various operations | processing, video, media |
| merge_videos | Merge multiple videos into one | video, merge, media |
| extract_frames | Extract frames from video | image, video, frames, media |
| extract_keyframes | Extract keyframes based on scene changes | image, video, frames, media |
| create_gif_preview | Create animated GIF from video | video, preview, media, gif |
| create_grid_preview | Create grid preview from video frames | video, grid, preview, media |
| create_video_montage | Create video montage from clips | video, montage, preview, media |

</details>

<details>
<summary>Research & Documents</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| arxiv_search_topic_tool | Search arXiv by topic/keywords | arxiv, research, papers, search, topic |
| arxiv_search_author_tool | Search arXiv by author | arxiv, research, papers, search, author |
| arxiv_search_year_tool | Search arXiv by publication year | arxiv, research, papers, search, year |
| arxiv_search_month_tool | Search arXiv by publication month | arxiv, research, papers, search, month |
| download_pdf_from_url | Download PDF from URL | document, pdf, download |
| pdf_tool | Perform operations on PDF files | document, split, merge, text, pdf |

</details>

<details>
<summary>Utilities</summary>

| Tool Name | Description | Tags |
|-----------|-------------|------|
| get_current_datetime | Get current date and time | datetime, current, utility |
| conclude_current_turn | Signal conclusion of current interaction | control, terminate, utility |

</details>

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Maintain backward compatibility

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE) 
