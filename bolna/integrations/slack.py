import os
from typing import Optional

from bolna.helpers.logger_config import configure_logger
from .base import PostCallContext, PostCallIntegration
from .runner import _post_with_retry

logger = configure_logger(__name__)

_MAX_SUMMARY_CHARS = 2800
_MAX_FIELD_VALUE_CHARS = 250


class SlackIntegration(PostCallIntegration):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    @classmethod
    def from_config(cls, config) -> "SlackIntegration":
        webhook_url = config.provider_config.webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            raise ValueError("slack webhook_url not provided in config or SLACK_WEBHOOK_URL env")
        return cls(webhook_url=webhook_url)

    async def execute(self, ctx: PostCallContext) -> None:
        payload = {"blocks": _build_slack_blocks(ctx)}
        await _post_with_retry(self.webhook_url, payload)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "n/a"
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _build_slack_blocks(ctx: PostCallContext) -> list:
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"call ended — {ctx.agent_name}"},
        },
    ]

    context_parts = []
    if ctx.call_sid:
        context_parts.append(f"*call_sid:* `{ctx.call_sid}`")
    context_parts.append(f"*duration:* {_format_duration(ctx.duration_seconds)}")
    if ctx.hangup_reason:
        context_parts.append(f"*hangup:* {ctx.hangup_reason}")
    context_parts.append(f"*run_id:* `{ctx.run_id}`")
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "  ·  ".join(context_parts)},
        }
    )

    if ctx.summary:
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*summary*\n{_truncate(ctx.summary, _MAX_SUMMARY_CHARS)}",
                },
            }
        )

    if ctx.extracted_data:
        fields = []
        for key, value in ctx.extracted_data.items():
            label = str(key).replace("_", " ")
            text = _truncate(str(value), _MAX_FIELD_VALUE_CHARS) if value not in (None, "") else "_n/a_"
            fields.append({"type": "mrkdwn", "text": f"*{label}*\n{text}"})
            if len(fields) == 10:
                break
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "fields": fields})

    if ctx.recording_url:
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "open recording"},
                        "url": ctx.recording_url,
                    }
                ],
            }
        )

    return blocks
