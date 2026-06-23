## Local docker setup

A basic local setup includes usage of [Twilio](local_setup/telephony_server/twilio_api_server.py) or [Plivo](local_setup/telephony_server/plivo_api_server.py) for telephony. We have dockerized the setup in `local_setup/`. One will need to populate an environment `.env` file from `.env.sample`.

The setup consists of four containers:

1. Telephony web server:
   * Choosing Twilio: for initiating the calls one will need to set up a [Twilio account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
   * Choosing Plivo: for initiating the calls one will need to set up a [Plivo account](https://www.plivo.com/)
2. Bolna server: for creating and handling agents 
3. `ngrok`: for tunneling. One will need to add the `authtoken` to `ngrok-config.yml`
4. `redis`: for persisting agents & prompt data

### Quick Start

The easiest way to get started is to use the provided script:

```bash
chmod +x start.sh
./start.sh
```

This script will check for Docker dependencies, build all services with BuildKit enabled, and start them in detached mode.

### Manual Setup

Alternatively, you can manually build and run the services:

1. Make sure you have Docker with Docker Compose V2 installed
2. Enable BuildKit for faster builds:
   ```bash
   export DOCKER_BUILDKIT=1
   export COMPOSE_DOCKER_CLI_BUILD=1
   ```
3. Build the images:
   ```bash
   docker compose build
   ```
4. Run the services:
   ```bash
   docker compose up -d
   ```

To run specific services only:

```bash
docker compose up -d bolna-app twilio-app
# or
docker compose up -d bolna-app plivo-app
```

Once the docker containers are up, you can now start to create your agents and instruct them to initiate calls.



## Example agents to create, use and start making calls
Go to the [Bolna examples](https://examples.bolna.dev/) to try out sample agents.
