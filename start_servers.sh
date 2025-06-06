#!/bin/bash

# Tmux session name
SESSION_NAME="knowledge-agent"

# Function to log messages
log() {
    local level=$1
    local message=$2
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message" | tee -a logs/startup.log
}

# Kill existing session if it exists, to ensure a clean start
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    log "INFO" "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

# Set PYTHONPATH to include project root
export PYTHONPATH=$PYTHONPATH:$(pwd)


log "INFO" "Creating new tmux session: $SESSION_NAME"

# Create a new detached tmux session and name the first window
tmux new-session -d -s $SESSION_NAME -n 'RAG Server'

# Send command to the first window (RAG Server)
tmux send-keys -t $SESSION_NAME:0 'source venv/bin/activate && export PYTHONPATH=$(pwd) && python3 src/app.py' C-m

# Create and start Bot Server in a new window
log "INFO" "Starting Bot server..."
tmux new-window -t $SESSION_NAME -n 'Bot Server'
tmux send-keys -t $SESSION_NAME:1 'source venv/bin/activate && export PYTHONPATH=$(pwd) && python3 src/mattermost_bot.py' C-m

# Create and start Ngrok in a new window
log "INFO" "Starting ngrok..."
tmux new-window -t $SESSION_NAME -n 'Ngrok'
tmux send-keys -t $SESSION_NAME:2 'ngrok start --log=stdout --log-level=debug mattermost-bot --config=ngrok.yml' C-m

log "INFO" "All services started in tmux session '$SESSION_NAME'."
echo
echo "Servers are starting up in the background inside a tmux session."
echo "You can attach to this session to see the live logs for each service."
echo
echo "To attach, run:"
echo "  tmux attach -t $SESSION_NAME"
echo
echo "Inside tmux, you can switch between windows with 'Ctrl+b' then 'n' (next) or 'p' (previous),"
echo "or 'Ctrl+b' then the window number (0, 1, 2)." 