## Local docker setup

A basic local setup includes usage of [Twilio](local_setup/telephony_server/twilio_api_server.py) or [Plivo](local_setup/telephony_server/plivo_api_server.py) for telephony. We have dockerized the setup in `local_setup/`. One will need to populate an environment `.env` file from `.env.sample`.

The setup consists of four containers:

1. Telephony web server:
   * Choosing Twilio: for initiating the calls one will need to set up a [Twilio account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
   * Choosing Plivo: for initiating the calls one will need to set up a [Plivo account](https://www.plivo.com/)
2. Bolna server: for creating and handling agents 
3. `ngrok`: for tunneling. One will need to add the `authtoken` to `ngrok-config.yml`
4. `redis`: for persisting agents & prompt data

Use docker to build the images using `.env` file as the environment file and run them locally for two services: `bolna-app` & telephony server app (`twilio-app` or `plivo-app`)
1. `docker-compose build --no-cache bolna-app <twilio-app | plivo-app>`: rebuild images for `twilio-app` or `plivo-app` as defined in the `docker-compose.yml`.
2. `docker-compose up bolna-app <twilio-app | plivo-app>`: run the build images

Once the docker containers are up, you can now start to create your agents and instruct them to initiate calls.



## Example agents to create, use and start making calls
Go to the [Bolna examples](https://examples.bolna.dev/) to try out sample agents.
