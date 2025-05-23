# Lecture Notes Bot for Mattermost

This bot allows users to query lecture notes using a RAG (Retrieval-Augmented Generation) pipeline through Mattermost.

## Project Structure

```
.
├── src/                    # Source code directory
│   ├── mattermost_bot.py   # Mattermost bot implementation
│   ├── app.py             # RAG pipeline server
│   └── ...                # Other source files
├── data/                  # Data directory for lecture notes
├── chroma_db/             # Vector database storage
├── requirements.txt       # Python dependencies
├── start_servers.sh      # Script to start all services
└── manage_ngrok_service.sh # Script to manage ngrok service
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Mattermost server with admin access
- Google API key for the RAG pipeline
- OpenAI API key for web search functionality

### 2. Environment Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in the required values:
   - `MATTERMOST_URL`: Your Mattermost server URL
   - `BOT_TOKEN`: Your Mattermost bot token
   - `GOOGLE_API_KEY`: Your Google API key
   - `OPENAI_API_KEY`: Your OpenAI API key

### 3. Mattermost Bot Setup

1. Log in to your Mattermost server as an administrator
2. Go to System Console > Integrations > Bot Accounts
3. Create a new bot account:
   - Username: lecture-bot
   - Display Name: Lecture Notes Bot
   - Description: Bot for querying lecture notes
4. Copy the generated bot token and add it to your `.env` file

### 4. Running the Bot

1. Start the RAG pipeline server:
   ```bash
   python src/app.py
   ```

2. Start the Mattermost bot server:
   ```bash
   python src/mattermost_bot.py
   ```

   Alternatively, you can use the provided script to start both services:
   ```bash
   ./start_servers.sh
   ```

3. The bot will be available in Mattermost using the `/lecture` command

## Usage

In any Mattermost channel, users can interact with the bot using:

```
@knowledge-agent <your question>
```

For example:
```
@knowledge-agent What is the definition of a vector norm?
```

The bot will respond with:
1. The answer to your question
2. Sources used to generate the answer
3. Any relevant context from the lecture notes

## Features

- Natural language querying of lecture notes
- Source attribution for answers
- Web search fallback when local context is insufficient
- Markdown formatting for better readability
- Error handling and user feedback

## Troubleshooting

If you encounter issues:

1. Check the logs in the `logs/` directory for error messages
2. Verify your API keys are correct in the `.env` file
3. Ensure the Mattermost bot has proper permissions
4. Check that the lecture notes are properly indexed in the RAG pipeline

## Security Notes

- Keep your API keys and bot tokens secure
- Do not commit the `.env` file to version control
- Regularly rotate your bot tokens
- Monitor bot usage and implement rate limiting if necessary 
