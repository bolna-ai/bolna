"""Every provider the config layer can name must resolve to a class here.

A provider offered in the catalog but missing from a registry fails only when a call actually
picks it, on that call, in production. The reverse — a registry key no enum names — is a handler
nothing can ever reach.
"""

import pytest

from bolna.enums import (
    LLMProvider,
    S2SProvider,
    SynthesizerProvider,
    TelephonyProvider,
    TranscriberProvider,
)
from bolna.providers import (
    SUPPORTED_INPUT_HANDLERS,
    SUPPORTED_INPUT_TELEPHONY_HANDLERS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_OUTPUT_HANDLERS,
    SUPPORTED_OUTPUT_TELEPHONY_HANDLERS,
    SUPPORTED_S2S_PROVIDERS,
    SUPPORTED_SYNTHESIZER_MODELS,
    SUPPORTED_TRANSCRIBER_PROVIDERS,
)

# Declared for the config layer's benefit; it names a source of numbers, not a media transport.
NON_TRANSPORT_TELEPHONY = {TelephonyProvider.DATABASE.value}

REGISTRIES = [
    ("synthesizer", SynthesizerProvider, SUPPORTED_SYNTHESIZER_MODELS),
    ("transcriber", TranscriberProvider, SUPPORTED_TRANSCRIBER_PROVIDERS),
    ("llm", LLMProvider, SUPPORTED_LLM_PROVIDERS),
    ("s2s", S2SProvider, SUPPORTED_S2S_PROVIDERS),
]


@pytest.mark.parametrize("label,enum,registry", REGISTRIES, ids=[r[0] for r in REGISTRIES])
def test_every_declared_provider_resolves_to_a_class(label, enum, registry):
    missing = sorted({member.value for member in enum} - set(registry))
    assert not missing, f"{label} providers with no class: {missing}"


@pytest.mark.parametrize("label,enum,registry", REGISTRIES, ids=[r[0] for r in REGISTRIES])
def test_no_registry_entry_is_unreachable(label, enum, registry):
    orphans = sorted(set(registry) - {member.value for member in enum})
    assert not orphans, f"{label} entries no enum names: {orphans}"


@pytest.mark.parametrize("registry", [SUPPORTED_INPUT_HANDLERS, SUPPORTED_OUTPUT_HANDLERS])
def test_every_telephony_transport_has_a_handler(registry):
    expected = {member.value for member in TelephonyProvider} - NON_TRANSPORT_TELEPHONY
    assert not sorted(expected - set(registry))


def test_a_carrier_that_can_listen_can_also_speak():
    """A carrier present in one direction only would take audio in and never play any back."""
    assert set(SUPPORTED_INPUT_TELEPHONY_HANDLERS) == set(SUPPORTED_OUTPUT_TELEPHONY_HANDLERS)


def test_the_carrier_subsets_stay_inside_the_full_handler_maps():
    assert set(SUPPORTED_INPUT_TELEPHONY_HANDLERS) <= set(SUPPORTED_INPUT_HANDLERS)
    assert set(SUPPORTED_OUTPUT_TELEPHONY_HANDLERS) <= set(SUPPORTED_OUTPUT_HANDLERS)
