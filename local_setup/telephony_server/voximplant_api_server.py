import os
import json
import requests
import uuid
from voximplant.apiclient import VoximplantAPI, VoximplantException
from dotenv import load_dotenv
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

app = FastAPI(port=8000)
load_dotenv()
port = 8000

rule_id = os.getenv('VOXIMPLANT_RULE_ID')


def populate_ngrok_tunnels():
    response = requests.get("http://ngrok:4040/api/tunnels")  # ngrok interface
    app_callback_url, websocket_url = None, None

    if response.status_code == 200:
        data = response.json()

        for tunnel in data['tunnels']:
            if tunnel['name'] == 'twilio-app':
                app_callback_url = tunnel['public_url']
            elif tunnel['name'] == 'voximplant':
                websocket_url = tunnel['public_url'].replace('https:', 'wss:')

        return app_callback_url, websocket_url
    else:
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")

@app.post('/call')
async def make_call(request: Request):
    try:
        call_details = await request.json()
        agent_id = call_details.get('agent_id', None)
        recipient_phone_number = call_details.get('recipient_phone_number')
        # recipient phone validation below
        if not recipient_phone_number:
            raise HTTPException(status_code=404, detail="Recipient phone number not provided")
        if not agent_id:
            raise HTTPException(status_code=404, detail="Agent not provided")
        
        if not call_details or "recipient_phone_number" not in call_details:
            raise HTTPException(status_code=404, detail="Recipient phone number not provided")

        app_callback_url, websocket_url = populate_ngrok_tunnels()

        print(f'app_callback_url: {app_callback_url}')
        print(f'websocket_url: {websocket_url}')
        print(f'recipient_phone_number: {recipient_phone_number}')

        voxapi = VoximplantAPI("credentials/voximplant.json")

        try:
            # Start the scenario
            res = voxapi.start_scenarios(
                rule_id=rule_id,  # Replace with your actual rule ID
                script_custom_data=json.dumps({
                    "ws_url": websocket_url,
                    "agent_id": agent_id,
                    "destination": recipient_phone_number
                })
            )
            print(res)
        except VoximplantException as e:
            raise HTTPException(status_code=500, detail=f"Error starting scenario: {e.message}")

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        print(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")