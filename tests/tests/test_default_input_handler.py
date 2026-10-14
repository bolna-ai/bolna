import asyncio

import pytest

from bolna.input_handlers.default import DefaultInputHandler


@pytest.mark.asyncio
async def test_malformed_message_does_not_stop_default_input_listener():
    input_queue = asyncio.Queue()
    queues = {
        "llm": asyncio.Queue(),
        "transcriber": asyncio.Queue(),
    }

    handler = DefaultInputHandler(
        queues=queues,
        queue=input_queue,
        input_types={"audio": 0},
        turn_based_conversation=True,
    )

    await handler.handle()

    await input_queue.put({"data": "missing type"})
    await input_queue.put({"type": "text", "data": "hello"})

    packet = await asyncio.wait_for(queues["llm"].get(), timeout=1)

    assert packet["data"] == "hello"

    handler.running = False
    handler.websocket_listen_task.cancel()
    try:
        await handler.websocket_listen_task
    except asyncio.CancelledError:
        pass
