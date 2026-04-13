import os
import json
import requests
import uuid
from dotenv import load_dotenv
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
import plivo

app = FastAPI()
load_dotenv()
port = 8002

plivo_auth_id = os.getenv("PLIVO_AUTH_ID")
plivo_auth_token = os.getenv("PLIVO_AUTH_TOKEN")
plivo_phone_number = os.getenv("PLIVO_PHONE_NUMBER")

# Initialize Plivo client
plivo_client = plivo.RestClient(os.getenv("PLIVO_AUTH_ID"), os.getenv("PLIVO_AUTH_TOKEN"))


def populate_ngrok_tunnels():
    response = requests.get("http://ngrok:4040/api/tunnels")  # ngrok interface
    telephony_url, bolna_url = None, None

    if response.status_code == 200:
        data = response.json()

        for tunnel in data["tunnels"]:
            if tunnel["name"] == "plivo-app":
                telephony_url = tunnel["public_url"]
            elif tunnel["name"] == "bolna-app":
                bolna_url = tunnel["public_url"].replace("https:", "wss:")

        return telephony_url, bolna_url
    else:
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")


@app.post("/call")
async def make_call(request: Request):
    try:
        call_details = await request.json()
        agent_id = call_details.get("agent_id", None)

        if not agent_id:
            raise HTTPException(status_code=404, detail="Agent not provided")

        if not call_details or "recipient_phone_number" not in call_details:
            raise HTTPException(status_code=404, detail="Recipient phone number not provided")

        telephony_host, bolna_host = populate_ngrok_tunnels()

        print(f"telephony_host: {telephony_host}")
        print(f"bolna_host: {bolna_host}")

        # adding hangup_url since plivo opens a 2nd websocket once the call is cut.
        # https://github.com/bolna-ai/bolna/issues/148#issuecomment-2127980509
        call = plivo_client.calls.create(
            from_=plivo_phone_number,
            to_=call_details.get("recipient_phone_number"),
            answer_url=f"{telephony_host}/plivo_connect?bolna_host={bolna_host}&agent_id={agent_id}",
            hangup_url=f"{telephony_host}/plivo_hangup_callback",
            answer_method="POST",
        )

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        print(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/plivo_connect")
async def plivo_connect(request: Request, bolna_host: str = Query(...), agent_id: str = Query(...)):
    try:
        bolna_websocket_url = f"{bolna_host}/chat/v1/{agent_id}"

        response = """
        <Response>
            <Stream bidirectional="true" keepCallAlive="true">{}</Stream>
        </Response>
        """.format(bolna_websocket_url)

        return PlainTextResponse(str(response), status_code=200, media_type="text/xml")

    except Exception as e:
        print(f"Exception occurred in plivo_connect: {e}")


@app.post("/plivo_hangup_callback")
async def plivo_hangup_callback(request: Request):
    # add any post call hangup processing
    return PlainTextResponse("", status_code=200)
