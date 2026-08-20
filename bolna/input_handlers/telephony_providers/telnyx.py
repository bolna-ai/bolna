from bolna.input_handlers.telephony import TelephonyInputHandler
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class TelnyxInputHandler(TelephonyInputHandler):
    """Telnyx Media Streaming (https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming)
    uses the same start/media/mark/stop event shape as Twilio, but nests the call id
    under "start" as call_control_id and carries the stream id at the top level of every
    frame as stream_id rather than inside "start"."""

    def __init__(
        self,
        queues,
        websocket=None,
        input_types=None,
        mark_event_meta_data=None,
        turn_based_conversation=False,
        is_welcome_message_played=False,
        observable_variables=None,
    ):
        super().__init__(
            queues,
            websocket,
            input_types,
            mark_event_meta_data,
            turn_based_conversation,
            is_welcome_message_played=is_welcome_message_played,
            observable_variables=observable_variables,
        )
        self.io_provider = "telnyx"

    async def call_start(self, packet):
        self.call_sid = packet["start"]["call_control_id"]
        self.stream_sid = packet["stream_id"]

    def get_mark_event_meta_data_obj(self, packet):
        mark_id = packet["mark"]["name"]
        return self.mark_event_meta_data.fetch_data(mark_id)
