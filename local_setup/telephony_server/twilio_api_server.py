import os
import json
import requests
import uuid
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from dotenv import load_dotenv
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

app = FastAPI()
load_dotenv()
port = 8001

# Request models
class CallRequest(BaseModel):
    agent_id: str
    recipient_phone_number: str

twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)


def populate_ngrok_tunnels():
    response = requests.get("http://ngrok:4040/api/tunnels")  # ngrok interface
    telephony_url, bolna_url = None, None

    if response.status_code == 200:
        data = response.json()

        for tunnel in data['tunnels']:
            if tunnel['name'] == 'twilio-app':
                telephony_url = tunnel['public_url']
            elif tunnel['name'] == 'bolna-app':
                bolna_url = tunnel['public_url'].replace('https:', 'wss:')

        return telephony_url, bolna_url
    else:
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")


@app.post('/call')
async def make_call(call_request: CallRequest):
    """
    Initiate a phone call using Twilio
    
    - **agent_id**: ID of the agent to use for the call
    - **recipient_phone_number**: Phone number to call (include country code, e.g., +1234567890)
    """
    try:
        agent_id = call_request.agent_id
        recipient_phone_number = call_request.recipient_phone_number

        telephony_host, bolna_host = populate_ngrok_tunnels()

        print(f'telephony_host: {telephony_host}')
        print(f'bolna_host: {bolna_host}')

        try:
            call = twilio_client.calls.create(
                to=recipient_phone_number,
                from_=twilio_phone_number,
                url=f"{telephony_host}/twilio_connect?bolna_host={bolna_host}&agent_id={agent_id}",
                method="POST",
                record=True
            )
        except Exception as e:
            print(f'make_call exception: {str(e)}')

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        print(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/twilio_connect')
async def twilio_connect(bolna_host: str = Query(...), agent_id: str = Query(...)):
    try:
        response = VoiceResponse()

        connect = Connect()
        bolna_websocket_url = f'{bolna_host}/chat/v1/{agent_id}'
        connect.stream(url=bolna_websocket_url)
        print(f"websocket connection done to {bolna_websocket_url}")
        response.append(connect)

        return PlainTextResponse(str(response), status_code=200, media_type='text/xml')

    except Exception as e:
        print(f"Exception occurred in twilio_callback: {e}")
