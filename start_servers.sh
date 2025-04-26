#!/bin/bash

# Configuration
MAX_RESTARTS=5
HEALTH_CHECK_INTERVAL=30
RESTART_DELAY=5
RAG_SERVER_PORT=5001
BOT_SERVER_PORT=5002

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to log messages
log() {
    local level=$1
    local message=$2
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message" | tee -a logs/startup.log
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null; then
        return 0
    fi
    return 1
}

# Function to check if a service is healthy
check_health() {
    local name=$1
    local port=$2
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$port/health > /dev/null; then
            return 0
        fi
        log "WARN" "Health check attempt $attempt/$max_attempts failed for $name"
        sleep 2
        attempt=$((attempt + 1))
    done
    return 1
}

# Function to check if server is already running and healthy
is_server_running() {
    local name=$1
    local port=$2
    
    if check_port $port; then
        if check_health "$name" "$port"; then
            log "INFO" "$name is already running and healthy on port $port"
            return 0
        else
            log "WARN" "$name is running but not healthy. Will attempt to restart."
            lsof -ti :$port | xargs kill -9
            sleep 1
            return 1
        fi
    fi
    return 1
}

# Function to start and monitor a server
start_server() {
    local name=$1
    local cmd=$2
    local port=$3
    local log_file="logs/${name}.log"
    local restart_count=0
    
    # Check if server is already running and healthy
    if is_server_running "$name" "$port"; then
        return 0
    fi

    log "INFO" "Starting $name server..."
    while [ $restart_count -lt $MAX_RESTARTS ]; do
        echo "$(date) - Starting $name (attempt $((restart_count + 1))/$MAX_RESTARTS)..." >> "$log_file"
        
        # Start the server
        PYTHONPATH=$PYTHONPATH:$(pwd) $cmd >> "$log_file" 2>&1 &
        local pid=$!
        
        # Wait for server to start
        sleep 5
        
        # Check if process is still running
        if ! ps -p $pid > /dev/null; then
            log "ERROR" "$name failed to start. Check $log_file for details."
            restart_count=$((restart_count + 1))
            sleep $RESTART_DELAY
            continue
        fi
        
        # Check health
        if ! check_health "$name" "$port"; then
            log "ERROR" "$name health check failed"
            kill $pid 2>/dev/null
            restart_count=$((restart_count + 1))
            sleep $RESTART_DELAY
            continue
        fi
        
        log "INFO" "$name started successfully"
        return $pid
    done
    
    if [ $restart_count -ge $MAX_RESTARTS ]; then
        log "ERROR" "$name failed to start after $MAX_RESTARTS attempts"
        return 1
    fi
}

# Function to handle cleanup
cleanup() {
    log "INFO" "Shutting down servers..."
    pkill -f "python3 mattermost/app.py"
    pkill -f "python3 mattermost/mattermost_bot.py"
    pkill -f "ngrok start"
    exit 0
}

# Trap signals
trap cleanup SIGINT SIGTERM

# Ensure virtual environment is activated if it exists
if [ -d "mattermost/venv" ]; then
    source mattermost/venv/bin/activate
    log "INFO" "Activated virtual environment"
fi

# Set PYTHONPATH to include project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start RAG server
log "INFO" "Starting RAG server..."
PYTHONPATH=$PYTHONPATH:$(pwd) python3 mattermost/app.py >> logs/rag.log 2>&1 &
RAG_PID=$!

# Wait for RAG server to initialize
sleep 10

# Check if RAG server is healthy
if ! check_health "RAG" $RAG_SERVER_PORT; then
    log "ERROR" "RAG server failed to initialize properly"
    cleanup
fi

# Start bot server
log "INFO" "Starting bot server..."
PYTHONPATH=$PYTHONPATH:$(pwd) python3 mattermost/mattermost_bot.py >> logs/bot.log 2>&1 &
BOT_PID=$!

# Wait for bot server to initialize
sleep 5

# Check if bot server is healthy
if ! check_health "Bot" $BOT_SERVER_PORT; then
    log "ERROR" "Bot server failed to initialize properly"
    cleanup
fi

# Start ngrok in the background
log "INFO" "Starting ngrok..."
ngrok start --log=stdout --log-level=debug mattermost-bot --config=ngrok.yml >> logs/ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to start
sleep 5
if ! ps -p $NGROK_PID > /dev/null; then
    log "ERROR" "ngrok failed to start. Check logs/ngrok.log for details."
    cleanup
fi

log "INFO" "All services are running!"
log "INFO" "Logs are available in the logs/ directory:"
log "INFO" "- RAG server: logs/rag.log"
log "INFO" "- Bot server: logs/bot.log"
log "INFO" "- ngrok: logs/ngrok.log"

# Keep the script running and monitoring
while true; do
    sleep $HEALTH_CHECK_INTERVAL
    
    # Check RAG server
    if ! ps -p $RAG_PID > /dev/null || ! check_health "RAG" $RAG_SERVER_PORT; then
        log "ERROR" "RAG server is down. Restarting..."
        PYTHONPATH=$PYTHONPATH:$(pwd) python3 mattermost/app.py >> logs/rag.log 2>&1 &
        RAG_PID=$!
        sleep 10
    fi
    
    # Check bot server
    if ! ps -p $BOT_PID > /dev/null || ! check_health "Bot" $BOT_SERVER_PORT; then
        log "ERROR" "Bot server is down. Restarting..."
        PYTHONPATH=$PYTHONPATH:$(pwd) python3 mattermost/mattermost_bot.py >> logs/bot.log 2>&1 &
        BOT_PID=$!
        sleep 5
    fi
    
    # Check ngrok
    if ! ps -p $NGROK_PID > /dev/null; then
        log "ERROR" "ngrok is down. Restarting..."
        ngrok start --log=stdout --log-level=debug mattermost-bot --config=ngrok.yml >> logs/ngrok.log 2>&1 &
        NGROK_PID=$!
        sleep 5
    fi
done 