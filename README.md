<h1 align="center">
</h1>
<p align="center">
  <p align="center"><b>End-to-end open-source voice agents platform</b>: Quickly build voice firsts conversational assistants through a json. </p>
</p>

<h4 align="center">
  <a href="https://discord.gg/59kQWGgnm8">Discord</a> |
  <a href="https://docs.bolna.dev">Hosted Docs</a> |
  <a href="https://bolna.dev">Website</a>
</h4>

<h4 align="center">
  <a href="https://discord.gg/59kQWGgnm8">
      <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
  </a>
  <a href="https://github.com/bolna-ai/bolna/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Bolna is released under the MIT license." />
  </a>
  <a href="https://github.com/bolna-ai/bolna/blob/main/CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs welcome!" />
  </a>
</h4>

> [!NOTE]
> We are actively looking for maintainers.

## Introduction

**[Bolna](https://bolna.dev)** is the end-to-end open source production ready framework for quickly building LLM based voice driven conversational applications.

## Demo
https://github.com/bolna-ai/bolna/assets/1313096/2237f64f-1c5b-4723-b7e7-d11466e9b226


## What is this repository?
This repository contains the entire orchestration platform to build voice AI applications. It technically orchestrates voice conversations using combination of different ASR+LLM+TTS providers and models over websockets.


## Components
Bolna helps you create AI Voice Agents which can be instructed to do tasks beginning with:

1. Orchestration platform (this open source repository)
2. Hosted APIs (https://docs.bolna.dev/api-reference/introduction) built on top of this orchestration platform [currently closed source]
3. No-code UI playground at https://playground.bolna.dev/ using the hosted APIs + tailwind CSS [currently closed source]


## Development philosophy
1. Any integration, enhancement or feature initially lands on this open source package since it forms the backbone of our Hosted APIs and dashboard
2. Post that we expose APIs or make changes to existing APIs as required for the same
3. Thirdly, we push it to the UI dashboard

```mermaid
graph LR;
    A[Bolna open source] -->B[Hosted APIs];
    B[Hosted APIs] --> C[Hosted Playground]
```

## Supported providers and models
1. Initiating a phone call using telephony providers like `Twilio`, `Plivo`, `Exotel` (coming soon), `Vonage` (coming soon) etc.
2. Transcribing the conversations using `Deepgram`, etc.
3. Using LLMs like `OpenAI`, `Llama`, `Cohere`, `Mistral`,  etc to handle conversations
4. Synthesizing LLM responses back to telephony using `AWS Polly`, `ElevenLabs`, `Deepgram`, `OpenAI`, `Azure`, `Cartesia`, `Smallest` etc.


Refer to the [docs](https://docs.bolna.dev/providers) for a deepdive into all supported providers.


## Local example setup [will be moved to a different repository]
A basic local setup includes usage of [Twilio](local_setup/telephony_server/twilio_api_server.py) or [Plivo](local_setup/telephony_server/plivo_api_server.py) for telephony. We have dockerized the setup in `local_setup/`. One will need to populate an environment `.env` file from `.env.sample`.

The setup consists of four containers:

1. Telephony web server:
   * Choosing Twilio: for initiating the calls one will need to set up a [Twilio account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
   * Choosing Plivo: for initiating the calls one will need to set up a [Plivo account](https://www.plivo.com/)
2. Bolna server: for creating and handling agents 
3. `ngrok`: for tunneling. One will need to add the `authtoken` to `ngrok-config.yml`
4. `redis`: for persisting agents & prompt data

Use docker to build the images using `.env` file as the environment file and run them locally
1. `docker-compose build --no-cache bolna-app <twilio-app | plivo-app>`: rebuild images
2. `docker-compose up bolna-app <twilio-app | plivo-app>`: run the build images

Once the docker containers are up, you can now start to create your agents and instruct them to initiate calls.


## Example agents to create, use and start making calls
You may try out different agents from [example.bolna.dev](https://examples.bolna.dev).


## Using your own providers
You can populate the `.env` file to use your own keys for providers.

<details>

<summary>ASR Providers</summary><br>
These are the current supported ASRs Providers:

| Provider     | Environment variable to be added in `.env` file |
|--------------|-------------------------------------------------|
| Deepgram     | `DEEPGRAM_AUTH_TOKEN`                           |

</details>
&nbsp;<br>

<details>
<summary>LLM Providers</summary><br>
Bolna uses LiteLLM package to support multiple LLM integrations.

These are the current supported LLM Provider Family:
https://github.com/bolna-ai/bolna/blob/477e08d6800dbf02931abeeea883d78451b7d7e2/bolna/providers.py#L29-L44

For LiteLLM based LLMs, add either of the following to the `.env` file depending on your use-case:<br><br>
`LITELLM_MODEL_API_KEY`: API Key of the LLM<br>
`LITELLM_MODEL_API_BASE`: URL of the hosted LLM<br>
`LITELLM_MODEL_API_VERSION`: API VERSION for LLMs like Azure

For LLMs hosted via VLLM, add the following to the `.env` file:<br>
`VLLM_SERVER_BASE_URL`: URL of the hosted LLM using VLLM

</details>
&nbsp;<br>

<details>

<summary>TTS Providers</summary><br>
These are the current supported TTS Providers:
https://github.com/bolna-ai/bolna/blob/c8a0d1428793d4df29133119e354bc2f85a7ca76/bolna/providers.py#L7-L14

| Provider   | Environment variable to be added in `.env` file  |
|------------|--------------------------------------------------|
| AWS Polly  | Accessed from system wide credentials via ~/.aws |
| Elevenlabs | `ELEVENLABS_API_KEY`                             |
| OpenAI     | `OPENAI_API_KEY`                                 |
| Deepgram   | `DEEPGRAM_AUTH_TOKEN`                            |
| Cartesia   | `CARTESIA_API_KEY`                            |
| Smallest   | `SMALLEST_API_KEY`                            |

</details>
&nbsp;<br>

<details>

<summary>Telephony Providers</summary><br>
These are the current supported Telephony Providers:

| Provider | Environment variable to be added in `.env` file                                                                                                                    |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Twilio   | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`|
| Plivo    | `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_PHONE_NUMBER`|

</details>

## Open-source v/s Hosted APIs
**We have in the past tried to maintain both the open source and the hosted solution (via APIs and a UI dashboard)**.

We have fluctuated b/w maintaining this repository purely from a point of time crunch and not interest.

Currently, we are continuing to maintain it for the community and improving the adoption of Voice AI.

Though the repository is completely open source, you can connect with us if interested in managed hosted offerings or more customized solutions.
<a href="https://calendly.com/bolna/30min"><img alt="Schedule a meeting" src="https://cdn.cookielaw.org/logos/122ecfc3-4694-42f1-863f-2db42d1b1e68/0bcbbcf4-9b83-4684-ba59-bc913c0d5905/c21bea90-f4f1-43d1-8118-8938bbb27a9d/logo.png" /></a>

## Extending with other Telephony Providers
In case you wish to extend and add some other Telephony like Vonage, Telnyx, etc. following the guidelines below:
1. Make sure bi-directional streaming is supported by the Telephony provider
2. Add the telephony-specific input handler file in [input_handlers/telephony_providers](https://github.com/bolna-ai/bolna/tree/master/bolna/input_handlers/telephony_providers) writing custom functions extending from the [telephony.py](https://github.com/bolna-ai/bolna/blob/master/bolna/input_handlers/telephony.py) class
   1. This file will mainly contain how different types of event packets are being ingested from the telephony provider
3. Add telephony-specific output handler file in [output_handlers/telephony_providers](https://github.com/bolna-ai/bolna/tree/master/bolna/output_handlers/telephony_providers) writing custom functions extending from the [telephony.py](https://github.com/bolna-ai/bolna/blob/master/bolna/output_handlers/telephony.py) class
   1. This mainly concerns converting audio from the synthesizer class to a supported audio format and streaming it over the websocket provided by the telephony provider
4. Lastly, you'll have to write a dedicated server like the example [twilio_api_server.py](https://github.com/bolna-ai/bolna/blob/master/local_setup/telephony_server/twilio_api_server.py) provided in [local_setup](https://github.com/bolna-ai/bolna/blob/master/local_setup/telephony_server) to initiate calls over websockets.


## Contributing
We love all types of contributions: whether big or small helping in improving this community resource.

1. There are a number of [open issues present](https://github.com/bolna-ai/bolna/issues) which can be good ones to start with
2. If you have suggestions for enhancements, wish to contribute a simple fix such as correcting a typo, or want to address an apparent bug, please feel free to initiate a new issue or submit a pull request
2. If you're contemplating a larger change or addition to this repository, be it in terms of its structure or the features, kindly begin by creating a new issue [open a new issue :octocat:](https://github.com/bolna-ai/bolna/issues/new) and outline your proposed changes. This will allow us to engage in a discussion before you dedicate a significant amount of time or effort. Your cooperation and understanding are appreciated
