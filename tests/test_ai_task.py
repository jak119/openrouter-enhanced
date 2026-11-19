"""Tests for the AI task entity without requiring Home Assistant."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch


def load_ai_task():
    def module(name, **attributes):
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        sys.modules[name] = stub
        return stub

    class HomeAssistantError(Exception):
        pass

    class OpenRouterAPIError(Exception):
        def __init__(self, message, status=None):
            super().__init__(message)
            self.status = status

    module("probatio", to_openapi=lambda structure, **kwargs: structure)
    module("homeassistant")
    module("homeassistant.components")
    module(
        "homeassistant.components.ai_task",
        AITaskEntity=type("AITaskEntity", (), {}),
        AITaskEntityFeature=SimpleNamespace(GENERATE_DATA=1, SUPPORT_ATTACHMENTS=2),
        GenDataTaskResult=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    module("homeassistant.components.conversation")
    module("homeassistant.config_entries", ConfigEntry=object)
    module("homeassistant.core", HomeAssistant=object)
    module("homeassistant.exceptions", HomeAssistantError=HomeAssistantError)
    module("homeassistant.helpers")
    module("homeassistant.helpers.llm", selector_serializer=object())
    module("homeassistant.helpers.entity_platform", AddConfigEntryEntitiesCallback=object)
    module("homeassistant.util", slugify=lambda value: value)
    module("homeassistant.util.json", json_loads=json.loads)
    module("_test_openrouter")
    module("_test_openrouter.api", OpenRouterAPIError=OpenRouterAPIError)
    module(
        "_test_openrouter.entity",
        OpenRouterLLMEntity=type("OpenRouterLLMEntity", (), {}),
        response_message=lambda response: response["choices"][0]["message"],
    )

    path = Path(__file__).parent.parent / "custom_components" / "openrouter_nativesdk" / "ai_task.py"
    spec = importlib.util.spec_from_file_location("_test_openrouter.ai_task", path)
    platform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(platform)
    return platform, HomeAssistantError, OpenRouterAPIError


class TestAITaskEntity(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.platform, self.error, self.api_error = load_ai_task()
        self.entity = self.platform.OpenRouterAITaskEntity()
        self.entity._async_build_payload = AsyncMock(return_value={"messages": [{"role": "user", "content": "instructions"}]})
        self.entity._async_chat = AsyncMock()
        self.chat_log = SimpleNamespace(llm_api=None, conversation_id="conversation-id")
        self.schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        self.task = SimpleNamespace(name="automation", structure=self.schema)

    async def test_structured_json_and_fenced_json(self):
        for text in ('{"name":"Lights"}', '```json\n{"name":"Lights"}\n```', ' \n```\r\n{"name":"Lights"}\r\n```\n'):
            with self.subTest(text=text):
                self.entity._async_build_payload.return_value = {"messages": [{"role": "user", "content": "instructions"}]}
                self.entity._async_chat.return_value = {"choices": [{"message": {"content": text}}]}
                result = await self.entity._async_generate_data(self.task, self.chat_log)
                self.assertEqual(result.data, {"name": "Lights"})
                self.assertEqual(result.conversation_id, "conversation-id")
                extra = self.entity._async_build_payload.call_args.kwargs
                self.assertEqual(extra["response_format"]["json_schema"]["schema"], self.schema)
                self.assertTrue(extra["provider"]["require_parameters"])
                self.assertIn("only JSON", self.entity._async_chat.call_args.args[0]["messages"][0]["content"])

    async def test_structured_instruction_extends_existing_system_message(self):
        messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "instructions"}]
        self.entity._async_build_payload.return_value = {"messages": messages}
        self.entity._async_chat.return_value = {"choices": [{"message": {"content": '{"name":"Lights"}'}}]}
        await self.entity._async_generate_data(self.task, self.chat_log)
        self.assertEqual(len(messages), 2)
        self.assertTrue(messages[0]["content"].startswith("System prompt\n\n"))
        self.assertIn("only JSON", messages[0]["content"])

    async def test_unstructured_text_is_unchanged(self):
        self.task.structure = None
        self.entity._async_chat.return_value = {"choices": [{"message": {"content": "```json\nnot JSON\n```"}}]}
        result = await self.entity._async_generate_data(self.task, self.chat_log)
        self.assertEqual(result.data, "```json\nnot JSON\n```")
        self.assertEqual(self.entity._async_build_payload.call_args.kwargs, {})
        self.assertEqual(len(self.entity._async_chat.call_args.args[0]["messages"]), 1)

    async def test_invalid_json_and_empty_response_are_logged(self):
        for text, error in (("```json\nnot JSON\n```", "invalid structured data"), ("", "no data"), ("private text", "invalid structured data")):
            with self.subTest(text=text):
                self.entity._async_chat.return_value = {"choices": [{"message": {"content": text}}]}
                with self.assertLogs(self.platform._LOGGER, level="ERROR") as logs:
                    with self.assertRaisesRegex(self.error, error):
                        await self.entity._async_generate_data(self.task, self.chat_log)
                self.assertIn(error, logs.output[0])
                self.assertNotIn("private text", logs.output[0])

    async def test_api_failure_is_logged_without_response_details(self):
        self.entity._async_chat.side_effect = self.api_error("private provider details", 429)
        with self.assertLogs(self.platform._LOGGER, level="ERROR") as logs:
            with self.assertRaisesRegex(self.error, "request failed"):
                await self.entity._async_generate_data(self.task, self.chat_log)
        self.assertIn("429", logs.output[0])
        self.assertNotIn("private provider details", logs.output[0])


if __name__ == "__main__":
    unittest.main()
