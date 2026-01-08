#!/bin/bash

# Configuration
IMAGE_NAME="hands-on-ml-custom:v1"
CONTAINER_NAME="my-jupyter-lab"
PORT=8888
# Mount the current directory to /home/jovyan/work
WORK_DIR="$PWD"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Check arguments
FORCE_BUILD=false
DOCKERFILE="Dockerfile"
USE_SURPRISE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            FORCE_BUILD=true
            log_info "Force build mode enabled."
            shift # past argument
            ;;
        --surprise)
            USE_SURPRISE=true
            shift # past argument
            ;;
        --surprise=*)
            SURPRISE_VALUE="${1#*=}"
            if [[ "$SURPRISE_VALUE" == "true" || "$SURPRISE_VALUE" == "1" || "$SURPRISE_VALUE" == "yes" ]]; then
                USE_SURPRISE=true
            elif [[ "$SURPRISE_VALUE" == "false" || "$SURPRISE_VALUE" == "0" || "$SURPRISE_VALUE" == "no" ]]; then
                USE_SURPRISE=false
            else
                log_error "Invalid value for --surprise: $SURPRISE_VALUE (expected true/false)"
                exit 1
            fi
            shift # past argument
            ;;
        -f|--file)
            DOCKERFILE="$2"
            log_info "Using Dockerfile: $DOCKERFILE"
            shift # past argument
            shift # past value
            ;;
        *)
            shift # past argument
            ;;
    esac
done

# Adjust Dockerfile based on surprise flag
if [ "$USE_SURPRISE" = true ]; then
    DOCKERFILE="Dockerfile_surprise"
    log_info "Surprise mode enabled. Using Dockerfile: $DOCKERFILE"
fi

# Adjust configuration based on Dockerfile
if [[ "$DOCKERFILE" == *"Dockerfile_surprise"* ]]; then
    IMAGE_NAME="hands-on-ml-surprise:v1"
    log_info "Switched to Surprise configuration: Image=$IMAGE_NAME, Container=$CONTAINER_NAME"
fi

# 2. Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker Desktop/Engine."
    exit 1
fi

# 2. Check Container Status
if [ "$FORCE_BUILD" = true ]; then
    log_info "Force build requested. Stopping and removing existing container..."
    docker stop $CONTAINER_NAME > /dev/null 2>&1
    docker rm $CONTAINER_NAME > /dev/null 2>&1
fi

log_info "Checking container status for '$CONTAINER_NAME'..."

# Get container status (running, exited, or empty if not exists)
CONTAINER_STATUS=$(docker inspect --format "{{.State.Status}}" $CONTAINER_NAME 2>/dev/null)
CURRENT_IMAGE=$(docker inspect --format "{{.Config.Image}}" $CONTAINER_NAME 2>/dev/null)

if [ -n "$CURRENT_IMAGE" ] && [ "$CURRENT_IMAGE" != "$IMAGE_NAME" ]; then
    log_info "Container image mismatch (current=$CURRENT_IMAGE, desired=$IMAGE_NAME). Recreating..."
    docker stop $CONTAINER_NAME > /dev/null 2>&1
    docker rm $CONTAINER_NAME > /dev/null 2>&1
    CONTAINER_STATUS=""
fi

if [ "$CONTAINER_STATUS" == "running" ]; then
    if [ "$FORCE_BUILD" = true ]; then
        log_warn "Container is running but force build requested. This should not happen (logic error)."
    else
        log_info "Container is already running."
    fi
    
elif [ "$CONTAINER_STATUS" == "exited" ] || [ "$CONTAINER_STATUS" == "created" ]; then
    log_info "Container exists but is stopped (Status: $CONTAINER_STATUS)."
    log_info "Starting container..."
    docker start $CONTAINER_NAME
    if [ $? -ne 0 ]; then
        log_error "Failed to start container."
        exit 1
    fi
    
else
    log_info "Container does not exist. Creating and starting..."
    
    # Check/Build image
    if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]] || [ "$FORCE_BUILD" = true ]; then
        if [ "$FORCE_BUILD" = true ]; then
             log_info "Forcing image rebuild..."
        else
             log_info "Image not found. Building custom Docker image $IMAGE_NAME..."
        fi
        
        if [ ! -f "$DOCKERFILE" ]; then
             log_error "$DOCKERFILE not found. Cannot build custom image."
             exit 1
        fi
        docker build -f "$DOCKERFILE" -t $IMAGE_NAME .
        
        if [ $? -ne 0 ]; then
            log_error "Docker build failed. Aborting."
            exit 1
        fi
    fi
    
    # Run container
    docker run -d \
        -p $PORT:8888 \
        -v "$WORK_DIR":/home/jovyan/work \
        -v "$WORK_DIR/fonts":/usr/share/fonts/truetype/custom_fonts:ro \
        -v "$WORK_DIR/config/matplotlibrc":/home/jovyan/.config/matplotlib/matplotlibrc:ro \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
        
    if [ $? -ne 0 ]; then
        log_error "Failed to create container."
        exit 1
    fi
fi

# 3. Refresh Font Cache
log_info "Refreshing font cache... this may take a few seconds"
docker exec -u 0 $CONTAINER_NAME fc-cache -fv > /dev/null 2>&1
# Clear matplotlib cache to force rebuild of font list
docker exec $CONTAINER_NAME rm -rf /home/jovyan/.cache/matplotlib

# 4. Retrieve Token and Show URL
log_info "Retrieving connection information..."
log_info "Waiting for JupyterLab to initialize (this may take up to 30s)..."

MAX_RETRIES=30
COUNT=0
TOKEN=""

while [ $COUNT -lt $MAX_RETRIES ]; do
    # 1. Try to get token via 'jupyter server list' (preferred)
    TOKEN_OUTPUT=$(docker exec $CONTAINER_NAME jupyter server list 2>/dev/null)
    TOKEN=$(echo "$TOKEN_OUTPUT" | grep -o 'token=[a-zA-Z0-9]*' | head -n 1 | cut -d= -f2)
    
    # 2. If not found, try logs
    if [ -z "$TOKEN" ]; then
        LOGS=$(docker logs $CONTAINER_NAME 2>&1)
        TOKEN=$(echo "$LOGS" | grep -o 'token=[a-zA-Z0-9]*' | head -n 1 | cut -d= -f2)
    fi

    # 3. If found, break loop
    if [ ! -z "$TOKEN" ]; then
        break
    fi

    ((COUNT++))
    echo -n "."
    sleep 1
done
echo ""

if [ -z "$TOKEN" ]; then
    log_error "Timed out waiting for token. Please check logs manually: docker logs $CONTAINER_NAME"
    exit 1
fi

URL="http://127.0.0.1:$PORT/lab?token=$TOKEN"

echo ""
echo "----------------------------------------------------------------"
echo -e "JupyterLab is running at:"
echo -e "${GREEN}$URL${NC}"
echo ""
echo "Work Directory: $WORK_DIR"
echo "Container Name: $CONTAINER_NAME"
echo "----------------------------------------------------------------"
