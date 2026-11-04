import asyncio
import unittest


def make_handler(input_types):
    from bolna.input_handlers.default import DefaultInputHandler
    queues = {"llm": asyncio.Queue(), "transcriber": asyncio.Queue()}
    handler = DefaultInputHandler.__new__(DefaultInputHandler)
    handler.queues = queues
    handler.input_types = input_types
    handler.turn_based_conversation = False
    return handler


class TestProcessTextUsesTextSequence(unittest.TestCase):

    def test_text_only_agent_does_not_raise_key_error(self):
        input_types = {"text": 1}
        handler = make_handler(input_types)
        try:
            handler._DefaultInputHandler__process_text("hello")
        except KeyError as e:
            self.fail(f"KeyError {e} raised for text-only agent")
        packet = handler.queues["llm"].get_nowait()
        self.assertEqual(packet["meta_info"]["sequence"], 1)

    def test_text_sequence_used_not_audio_sequence(self):
        input_types = {"audio": 99, "text": 42}
        handler = make_handler(input_types)
        handler._DefaultInputHandler__process_text("world")
        packet = handler.queues["llm"].get_nowait()
        self.assertEqual(packet["meta_info"]["sequence"], 42)

    def test_packet_type_is_text(self):
        input_types = {"text": 5}
        handler = make_handler(input_types)
        handler._DefaultInputHandler__process_text("test")
        packet = handler.queues["llm"].get_nowait()
        self.assertEqual(packet["meta_info"]["type"], "text")

    def test_packet_data_is_correct(self):
        input_types = {"text": 0}
        handler = make_handler(input_types)
        handler._DefaultInputHandler__process_text("exact text")
        packet = handler.queues["llm"].get_nowait()
        self.assertEqual(packet["data"], "exact text")


if __name__ == "__main__":
    unittest.main()
