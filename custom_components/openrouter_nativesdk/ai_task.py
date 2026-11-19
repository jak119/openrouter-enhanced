"""AI task platform."""
from __future__ import annotations

import json
import logging
import re

import probatio
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import slugify
from homeassistant.util.json import json_loads

from .api import OpenRouterAPIError
from .entity import OpenRouterLLMEntity, response_message

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    for subentry in entry.subentries.values():
        if subentry.subentry_type == "ai_task_data":
            async_add_entities([OpenRouterAITaskEntity(entry, subentry)], config_subentry_id=subentry.subentry_id)


class OpenRouterAITaskEntity(OpenRouterLLMEntity, ai_task.AITaskEntity):
    """OpenRouter structured-data entity."""

    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS

    async def _async_generate_data(self, task: ai_task.GenDataTask, chat_log: conversation.ChatLog) -> ai_task.GenDataTaskResult:
        try:
            return await self._async_generate_data_response(task, chat_log)
        except OpenRouterAPIError as err:
            _LOGGER.error("OpenRouter data-generation request failed (HTTP status: %s)", err.status)
            raise HomeAssistantError(f"OpenRouter data-generation request failed: {err}") from err
        except HomeAssistantError as err:
            _LOGGER.error("OpenRouter data generation failed: %s", err)
            raise
        except Exception as err:
            _LOGGER.error("OpenRouter data generation failed (%s)", type(err).__name__)
            raise

    async def _async_generate_data_response(self, task: ai_task.GenDataTask, chat_log: conversation.ChatLog) -> ai_task.GenDataTaskResult:
        extra = {}
        if task.structure is not None:
            serializer = chat_log.llm_api.custom_serializer if chat_log.llm_api else llm.selector_serializer
            schema = probatio.to_openapi(task.structure, custom_serializer=serializer, openapi_version="3.1.0")
            extra = {
                "response_format": {"type": "json_schema", "json_schema": {"name": slugify(task.name)[:64] or "response", "schema": schema}},
                # Only route to endpoints that honour response_format.
                "provider": {"require_parameters": True},
            }
        payload = await self._async_build_payload(chat_log, **extra)
        if task.structure is not None:
            payload.setdefault("plugins", []).append({"id": "response-healing"})
            instruction = "Respond with only JSON matching the requested schema, without Markdown."
            messages = payload["messages"]
            if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
                messages[0]["content"] += f"\n\n{instruction}"
            else:
                messages.insert(0, {"role": "system", "content": instruction})
        response = await self._async_chat(payload)
        text = response_message(response).get("content")
        if not text:
            raise HomeAssistantError("OpenRouter returned no data")
        if task.structure is None:
            return ai_task.GenDataTaskResult(conversation_id=chat_log.conversation_id, data=text)
        try:
            if isinstance(text, str) and (fenced := re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*\s*", text, re.IGNORECASE | re.DOTALL)):
                text = fenced.group(1)
            data = json_loads(text)
        except (json.JSONDecodeError, TypeError) as err:
            raise HomeAssistantError("OpenRouter returned invalid structured data") from err
        return ai_task.GenDataTaskResult(conversation_id=chat_log.conversation_id, data=data)
