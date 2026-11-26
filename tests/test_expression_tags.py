"""Expressive tags: how a reply's neutral tags are read, checked against a voice, and removed."""

import re

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
    for model in ("sonic-3", "sonic-3.5", "sonic-3.6"):
        sonic = tags.capabilities_for("cartesia", model)
        assert sonic.pause_ms and {"slow", "fast", "spell"} <= sonic.deliveries and sonic.sounds == {"laugh"}
        assert {"break", "speed", "spell"} <= sonic.raw_xml and sonic.drops_brackets and not sonic.moods
    for model in ("eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_multilingual_v2"):
        ssml = tags.capabilities_for("elevenlabs", model)
        assert ssml.pause_ms and ssml.raw_xml == {"break"} and not ssml.moods and not ssml.sounds
        assert not ssml.raw_brackets
    v4 = tags.capabilities_for("elevenlabs", "eleven_v4_turbo")
    assert v4.raw_brackets and {"slow", "fast", "stress"} <= v4.deliveries and v4.sounds == {"laugh", "sigh"}
    assert v4.moods == {"warm", "calm", "apologetic", "sympathetic", "excited"}
    v3 = tags.capabilities_for("elevenlabs", "eleven_v3_conversational")
    assert "fast" not in v3.deliveries and v3.sounds == {"laugh"}  # v3 performs neither [quickly] nor [sighs]
    # A model without confirmed tag support, or a provider without tags: words only.
    assert tags.capabilities_for("cartesia", "sonic-4") == tags.NO_TAGS
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
    assert tags.remove_tags("Your PIN is <<spell: 4821") == "Your PIN is 4821"
    assert tags.remove_tags("Read <<slow: 98 76 54") == "Read 98 76 54"
    assert tags.split_unfinished("Ok [[pause 1.5s", final=True) == ("Ok ", "")


def test_a_malformed_tag_or_a_bracketed_word_keeps_its_words():
    assert tags.words_only(tags.segments("ID <<spell BK4821>> ok.", tags.NO_TAGS)) == "ID BK4821 ok."
    assert tags.words_only(tags.segments("Read <<slow 98 76 54>>.", tags.NO_TAGS)) == "Read 98 76 54."
    assert tags.words_only(tags.segments("Choose [Gold] or [Silver].", tags.NO_TAGS)) == "Choose Gold or Silver."
    assert tags.RawTag("[Gold]") in tags.segments("Choose [Gold].", tags.Capabilities(raw_brackets=True))


def test_a_bracketed_tag_word_is_removed_but_other_bracketed_words_are_kept():
    text = "[warmly] Hello [laughs] ok [Gold] [Short Pause] fine [French accent] hi [Silver]."
    assert tags.words_only(tags.segments(text, tags.NO_TAGS)) == " Hello ok Gold fine hi Silver."


def test_delivery_directions_agents_write_are_tags_however_they_are_combined():
    text = (
        "[smiling] Hi [warm and friendly] there [warm, cheerful, fast] ok [gently reassuring] [calm and steady] "
        "[lightly curious] [casual, rapid] [energetic] [starts laughing] [strong Hindi accent] done"
    )
    assert tags.words_only(tags.segments(text, tags.NO_TAGS)) == " Hi there ok done"


def test_the_providers_documented_tags_are_tag_words():
    documented = (
        "laughs harder, wheezing, whispers, sighs, exhales, sarcastic, mischievously, applause, clears throat, "
        "inhales deeply, annoyed, appalled, deadpan, sheepishly, content, serene, melancholic, skeptical, determined"
    )
    assert all(tags.is_expressive_word(word) for word in documented.split(", "))


def test_placeholders_and_real_text_in_brackets_keep_their_words():
    text = "Hi [name], your [plan name] for the [water purifier] is ready, [sir]. [call ended]"
    assert tags.words_only(tags.segments(text, tags.NO_TAGS)) == text.replace("[", "").replace("]", "")


def test_cartesia_drops_every_bracket_it_does_not_perform_as_the_voice_itself_would():
    sonic = tags.capabilities_for("cartesia", "sonic-3.6")
    kept = tags.segments("[warm and friendly] Pick [Gold] now [laughter]", sonic)
    assert tags.words_only(kept) == " Pick now "
    assert tags.RawTag("[laughter]") in kept


def test_a_raw_break_is_a_pause_on_a_voice_that_pauses_another_way():
    v4 = tags.capabilities_for("elevenlabs", "eleven_v4_turbo")
    assert tags.segments('Hi <break time="1.5s"/> there', v4) == [
        tags.Words("Hi "),
        tags.Pause(1500),
        tags.Words(" there"),
    ]
    assert tags.segments('Hi <break time="300ms" /> there', v4)[1] == tags.Pause(300)


def test_rules_spell_ids_and_slow_phone_numbers_and_otps():
    text = "Booking BK48217 is done. Car KA01AB1234. Call 9876543210 or +919876543210. Your OTP is 482913. आपका ओटीपी 4829 है।"
    kept = tags.segments(text, tags.capabilities_for("cartesia", "sonic-3.6"), rules=True)
    marked = {(s.text, s.delivery) for s in kept if isinstance(s, tags.Words) and s.delivery}
    assert marked == {
        ("BK48217", "spell"),
        ("KA01AB1234", "spell"),
        ("9876543210", "slow"),
        ("+919876543210", "slow"),
        ("482913", "slow"),
        ("4829", "slow"),
    }


def test_rules_leave_amounts_years_units_and_product_names_alone():
    text = (
        "In 2024 we sold 15000 units: 100MB, 2BHK, 4K, iPhone15, COVID19, 24X7 and Rs 1,25,000. Pin code 560001 ok. "
        "Use code SAVE10, GET50, WELCOME20 or FESTIVE100 at checkout."  # promo codes are said as words, not letters
    )
    kept = tags.segments(text, tags.capabilities_for("cartesia", "sonic-3.6"), rules=True)
    assert [s for s in kept if isinstance(s, tags.Words) and s.delivery] == [tags.Words("560001", "slow")]


def test_rules_never_touch_text_inside_a_tag():
    kept = tags.segments("<<slow: BK48217>> ok", tags.capabilities_for("cartesia", "sonic-3.6"), rules=True)
    assert kept[0] == tags.Words("BK48217", "slow")


REPLY = "<<apologetic>> Sorry. [[pause 400]] Your ID <<spell: BK4821>> is <<slow: 98765 43210>> ok ^^now^^ [[laugh]]"


def _format(provider, model, fn, **kw):
    return fn(tags.segments(REPLY, tags.capabilities_for(provider, model)), **kw)


def test_cartesia_gets_its_own_ssml_with_speed_back_to_the_agents_speed():
    assert _format("cartesia", "sonic-3.6", tags.format_cartesia, speed=1.25) == (
        ' Sorry. <break time="400ms"/> Your ID <spell>BK4821</spell> is <speed ratio="1.0"/> 98765 43210 '
        '<speed ratio="1.25"/> ok now [laughter] '
    )


def test_elevenlabs_v2_gets_ssml_breaks_and_spelled_out_codes():
    assert _format("elevenlabs", "eleven_turbo_v2_5", tags.format_eleven_ssml) == (
        ' Sorry. <break time="0.4s" /> Your ID B K 4 8 2 1 is 98765 43210 ok now '
    )


def test_elevenlabs_v3_and_v4_get_audio_tags():
    assert _format("elevenlabs", "eleven_v4_turbo", tags.format_audio_tags) == (
        "[apologetic] Sorry. [short pause] Your ID B K 4 8 2 1 is [slowly] 98765 43210 ok NOW [laughs] "
    )
    long_pause = tags.segments("A [[pause 1500]] B <<warm>> C", tags.capabilities_for("elevenlabs", "eleven_v4_turbo"))
    assert tags.format_audio_tags(long_pause) == "A [long pause] B [warmly] C"


def test_a_slow_or_fast_span_stays_inside_cartesias_speed_range():
    sonic = tags.capabilities_for("cartesia", "sonic-3.6")
    slow = tags.format_cartesia(tags.segments("<<slow: 1 2 3>>", sonic), speed=0.7)
    fast = tags.format_cartesia(tags.segments("<<fast: 1 2 3>>", sonic), speed=1.4)
    assert '<speed ratio="0.6"/>' in slow and '<speed ratio="1.5"/>' in fast


def test_the_prompt_guide_offers_only_what_the_voice_performs():
    v4 = tags.prompt_guide(tags.capabilities_for("elevenlabs", "eleven_v4_turbo"))
    assert "[[pause" in v4 and "<<slow:" in v4 and "<<warm>>" in v4 and "[[laugh]]" in v4
    turbo = tags.prompt_guide(tags.capabilities_for("elevenlabs", "eleven_turbo_v2_5"))
    assert "[[pause" in turbo and "<<warm>>" not in turbo and "[[laugh]]" not in turbo and "<<slow:" not in turbo
    assert tags.prompt_guide(tags.NO_TAGS) == ""


def test_every_voice_gets_an_example_with_only_the_tags_it_performs():
    for provider, model in (
        ("cartesia", "sonic-3"),
        ("cartesia", "sonic-3.6"),
        ("elevenlabs", "eleven_turbo_v2_5"),
        ("elevenlabs", "eleven_v3_conversational"),
        ("elevenlabs", "eleven_v4_turbo"),
    ):
        caps = tags.capabilities_for(provider, model)
        guide = tags.prompt_guide(caps)
        assert "\nExample: " in guide, model
        for s in tags.parse(guide):
            if isinstance(s, tags.Pause):
                assert caps.pause_ms, model
            elif isinstance(s, tags.Mood):
                assert s.name in caps.moods, (model, s)
            elif isinstance(s, tags.Sound):
                assert s.name in caps.sounds, (model, s)
            elif isinstance(s, tags.Words) and s.delivery:
                assert s.delivery in caps.deliveries, (model, s)


def test_the_prompt_guide_says_which_moment_each_mood_is_for():
    v4 = tags.capabilities_for("elevenlabs", "eleven_v4_turbo")
    guide = tags.prompt_guide(v4)
    for mood in v4.moods:
        assert re.search(rf"<<{mood}>> [a-z]", guide), mood


def test_the_transcript_keeps_bracketed_text_that_is_not_a_tag_and_no_gap_where_a_tag_was():
    assert tags.remove_tags("Pick [Gold] now [smiling]. Thanks [warm and friendly]!") == "Pick Gold now. Thanks!"


def test_crafted_text_cannot_stall_the_parser():
    import time

    crafted = [
        "<<a" + " " * 300 + ":" + "\n" * 600 + "<>>",
        "<<a:" + " " * 9990 + "<>>",
        "<<a" + " " * 5000 + "x" * 4000,
        "[[pause 5" + " " * 9000 + "x",
        "[[laugh" + " " * 9000 + "]",
        "<<warm>> Hello" + "\n" * 40000 + "there",  # a long run of line breaks, no punctuation after it
        "[[pause 1]] Hi" + " \n" * 20000 + "x",
        "[[" + " " * 40000,
    ]
    for text in crafted:
        started = time.perf_counter()
        tags.remove_tags(text)
        tags.has_tags(text)
        assert time.perf_counter() - started < 0.5, text[:12]


def test_a_huge_pause_value_is_clamped_not_a_crash():
    assert tags.parse("[[pause " + "9" * 400 + "]]") == [tags.Pause(60000)]
    huge_break = '<break time="' + "9" * 400 + 's"/>'
    v4 = tags.capabilities_for("elevenlabs", "eleven_v4_turbo")
    assert tags.segments(f"Hi {huge_break} there", v4)[1] == tags.Pause(3000)
    assert tags.remove_tags("Hi [[pause " + "9" * 400 + "]] there") == "Hi there"


def test_a_tag_an_interruption_cut_off_is_removed_from_the_saved_text():
    assert tags.remove_tags("Your OTP is <<slow: 4 7") == "Your OTP is 4 7"
    assert tags.remove_tags("Sure [[pause") == "Sure"
    assert tags.remove_tags("It is ^^really") == "It is really"


def test_text_without_tags_is_saved_exactly_as_written():
    for text in ("हो गया है ।", "Odds are 3 : 1 ", "Press [1] now"):
        assert tags.remove_tags(text) == text


def test_a_spell_the_reply_already_wrote_is_not_wrapped_again():
    sonic = tags.capabilities_for("cartesia", "sonic-3.6")
    sent = tags.format_cartesia(tags.segments("Your ID is <spell>BK48217</spell>.", sonic, rules=True))
    assert sent.count("<spell>") == 1 and sent.count("</spell>") == 1


def test_a_sound_at_the_very_end_of_a_reply_is_dropped_on_elevenlabs_v3_and_v4():
    v4 = tags.capabilities_for("elevenlabs", "eleven_v4_turbo")
    last = tags.segments("<<warm>> Done. [[pause 400]] See you Monday. [[laugh]] ", v4, end_of_reply=True)
    assert not any(isinstance(s, tags.Sound) for s in last)
    raw = tags.segments("[warmly] Done. [long pause] See you. [laughs] [sighs]", v4, end_of_reply=True)
    assert tags.format_audio_tags(raw) == "[warmly] Done. [long pause] See you. "
    # Only at the very end of the reply: a laugh mid-reply, or in a piece that isn't the last, stays.
    assert tags.Sound("laugh") in tags.segments("Ha! [[laugh]] That was fun.", v4, end_of_reply=True)
    assert tags.Sound("laugh") in tags.segments("See you. [[laugh]]", v4, end_of_reply=False)
    # Cartesia plays a trailing laugh fine, so it keeps it.
    sonic = tags.capabilities_for("cartesia", "sonic-3.6")
    assert tags.Sound("laugh") in tags.segments("See you. [[laugh]]", sonic, end_of_reply=True)
