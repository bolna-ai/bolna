"""Expressive tags: how a reply's neutral tags are read, checked against a voice, and removed."""

from bolna.helpers import expression_tags as tags


def test_remove_tags_leaves_only_the_words():
    text = "<<apologetic>> Sorry for the wait. [[pause 400]] Your ID is <<spell: BK4821>>, ^^today^^ [[laugh]] only."
    assert tags.remove_tags(text) == "Sorry for the wait. Your ID is BK4821, today only."


VOICE = tags.Capabilities(
    pause_ms=(100, 3000),
    deliveries=frozenset({"slow", "spell"}),
    moods=frozenset({"warm", "apologetic"}),
    sounds=frozenset({"laugh"}),
)


def test_a_voice_keeps_only_the_tags_it_supports():
    text = "<<apologetic>> Sorry. <<thrilled>> [[pause 400]] Code <<spell: BK4821>>, ^^now^^ [[sigh]] [[laugh]]"
    assert tags.segments(text, VOICE) == [
        tags.Mood("apologetic"),
        tags.Words(" Sorry. "),
        tags.Words(" "),
        tags.Pause(400),
        tags.Words(" Code "),
        tags.Words("BK4821", "spell"),
        tags.Words(", "),
        tags.Words("now"),  # stress isn't supported by this voice: the words stay, plain
        tags.Words(" "),
        tags.Words(" "),
        tags.Sound("laugh"),
    ]


def test_a_voice_without_tag_support_gets_words_only():
    text = "<<warm>> Hi [[pause]] there [[laugh]]"
    assert tags.words_only(tags.segments(text, tags.NO_TAGS)) == " Hi there "


def test_pauses_are_clamped_and_never_back_to_back():
    text = "A [[pause 50]] B [[pause 9000]] C [[pause]] [[pause 700]] D"
    assert [s for s in tags.segments(text, VOICE) if isinstance(s, tags.Pause)] == [
        tags.Pause(100),
        tags.Pause(3000),
        tags.Pause(500),  # the [[pause 700]] right after it is dropped
    ]


def test_a_reply_gets_at_most_the_pauses_left_for_it():
    text = "A [[pause]] B [[pause]] C [[pause]] D"
    kept = tags.segments(text, VOICE, pauses_left=2)
    assert sum(isinstance(s, tags.Pause) for s in kept) == 2
    assert tags.words_only(kept) == "A B C D"


def test_mood_is_kept_only_in_the_voices_mood_languages():
    english_mood = tags.Capabilities(moods=frozenset({"warm"}), mood_languages=frozenset({"en"}))
    assert tags.segments("<<warm>> Hi", english_mood, language="en") == [tags.Mood("warm"), tags.Words(" Hi")]
    assert tags.segments("<<warm>> नमस्ते", english_mood, language="hi") == [tags.Words(" नमस्ते")]


def test_spell_is_kept_only_for_ids_and_emails():
    assert tags.segments("<<spell: support@bolna.ai>>", VOICE) == [tags.Words("support@bolna.ai", "spell")]
    assert tags.segments("<<spell: your order>>", VOICE) == [tags.Words("your order")]


def test_raw_provider_tags_are_removed_unless_the_voice_performs_them():
    text = 'Hi <break time="1s"/> there [calm] ok, <5 minutes [1]'
    assert tags.words_only(tags.segments(text, tags.NO_TAGS)) == "Hi there ok, <5 minutes [1]"
    breaks_only = tags.Capabilities(raw_xml=frozenset({"break"}))
    assert tags.RawTag('<break time="1s"/>', "break") in tags.segments(text, breaks_only)
    brackets = tags.Capabilities(raw_brackets=True)
    assert tags.RawTag("[calm]") in tags.segments(text, brackets)


def test_a_tag_a_piece_ends_inside_is_held_back():
    assert tags.split_unfinished("Sorry for the wait. [[pause") == ("Sorry for the wait. ", "[[pause")
    assert tags.split_unfinished("Code <<spell: BK") == ("Code ", "<<spell: BK")
    assert tags.split_unfinished('Hi <break time="1') == ("Hi ", '<break time="1')
    assert tags.split_unfinished("say ^^now") == ("say ", "^^now")
    assert tags.split_unfinished("Done [[pause 400]] <<warm>> ok") == ("Done [[pause 400]] <<warm>> ok", "")
    assert tags.split_unfinished("if a < b then [1]") == ("if a < b then [1]", "")


def test_a_tag_that_never_closes_loses_its_head_but_keeps_its_words():
    assert tags.split_unfinished("Thanks. [[pause 400", final=True) == ("Thanks. ", "")
    assert tags.split_unfinished("<<warm Sorry for the wait.", final=True) == (" Sorry for the wait.", "")
    assert tags.split_unfinished("Hi ^^now", final=True) == ("Hi now", "")
    assert tags.split_unfinished('Hi <break time="1', final=True) == ("Hi ", "")
    too_long = "<<slow: " + "word " * 20
    assert tags.split_unfinished(too_long) == (" " + "word " * 20, "")


def test_releasing_one_open_tag_still_catches_the_next():
    assert tags.split_unfinished("<<warm Sorry [[pause", final=True) == (" Sorry ", "")


def test_voices_get_only_their_confirmed_tags():
    sonic = tags.capabilities_for("cartesia", "sonic-3.6")
    assert sonic.pause_ms and {"slow", "spell"} <= sonic.deliveries and "break" in sonic.raw_xml
    turbo = tags.capabilities_for("elevenlabs", "eleven_turbo_v2_5")
    assert turbo.pause_ms and "break" in turbo.raw_xml and not turbo.moods and not turbo.raw_brackets
    v4 = tags.capabilities_for("elevenlabs", "eleven_v4_turbo")
    assert v4.pause_ms and v4.raw_brackets and not v4.raw_xml
    # A model without confirmed tag support, or a provider without tags: words only.
    assert tags.capabilities_for("cartesia", "sonic-3") == tags.NO_TAGS
    assert tags.capabilities_for("elevenlabs", "eleven_multilingual_v2") == tags.NO_TAGS
    assert tags.capabilities_for("sarvam", "bulbul:v3") == tags.NO_TAGS


def test_every_tag_spelling_is_handled_never_left_as_text():
    text = "A [[pause 0.5s]] B [[pause 400ms]] C <<soft tone>> D [[clears throat]] E <<tone: calm>> F"
    assert tags.words_only(tags.segments(text, tags.NO_TAGS)) == "A B C D E calm F"
    assert [s for s in tags.segments(text, VOICE) if isinstance(s, tags.Pause)] == [tags.Pause(500), tags.Pause(400)]


def test_stray_angle_or_square_brackets_never_cost_the_reply_its_words():
    assert tags.split_unfinished("Call us if x<y then press one.", final=True) == ("Call us if x<y then press one.", "")
    assert tags.split_unfinished("Press [Enter to continue now", final=True) == ("Press Enter to continue now", "")
    assert tags.split_unfinished('Hi <break time="1', final=True) == ("Hi ", "")


def test_a_flood_of_openers_neither_crashes_nor_loses_words():
    assert tags.split_unfinished("<<" * 3000, final=True) == ("", "")
    assert tags.split_unfinished("[a" * 3000, final=True) == ("a" * 3000, "")


def test_a_tag_that_never_closes_keeps_its_numbers():
    assert tags.remove_tags(tags.split_unfinished("Your PIN is <<spell: 4821", final=True)[0]) == "Your PIN is 4821"
    assert tags.remove_tags(tags.split_unfinished("Read <<slow: 98 76 54", final=True)[0]) == "Read 98 76 54"
    assert tags.split_unfinished("Ok [[pause 1.5s", final=True) == ("Ok ", "")


def test_a_malformed_tag_or_a_bracketed_word_keeps_its_words():
    assert tags.words_only(tags.segments("ID <<spell BK4821>> ok.", tags.NO_TAGS)) == "ID BK4821 ok."
    assert tags.words_only(tags.segments("Read <<slow 98 76 54>>.", tags.NO_TAGS)) == "Read 98 76 54."
    assert tags.words_only(tags.segments("Choose [Gold] or [Silver].", tags.NO_TAGS)) == "Choose Gold or Silver."
    assert tags.RawTag("[Gold]") in tags.segments("Choose [Gold].", tags.Capabilities(raw_brackets=True))


def test_a_bracketed_tag_word_is_removed_but_other_bracketed_words_are_kept():
    text = "[warmly] Hello [laughs] ok [Gold] [Short Pause] fine [French accent] hi [Silver]."
    assert tags.words_only(tags.segments(text, tags.NO_TAGS)) == " Hello ok Gold fine hi Silver."
