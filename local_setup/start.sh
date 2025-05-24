
set -e

set -x

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "Docker Compose (v2 CLI) is not available. Please install or update Docker."
    exit 1
fi

# Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

cd "$(dirname "$0")"

echo "🔧 Building services using Docker Compose with BuildKit enabled..."

# Build all services defined in docker-compose.yml
docker compose build

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! Starting the services..."
else
    echo "Build failed. Please check the error messages above."
    exit 1
fi

docker compose up -d

echo "Services are up and running!"
echo "Use 'docker compose ps' to see running containers."
echo "Use 'docker compose logs -f' to view logs."
