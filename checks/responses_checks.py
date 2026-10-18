import json
import unittest

from app.agent_gateway.responses import convert_request, convert_answer, output_events
from app.agent_gateway.protocol import ProtocolError


class ResponsesChecks(unittest.TestCase):
    def test_function_roundtrip_preserves_ids_and_result(self):
        body = {"input": [{"role": "user", "content": "Read the file"},
                          {"type": "function_call", "name": "shell_command", "call_id": "call_a", "arguments": '{"command":"cat a"}'},
                          {"type": "function_call_output", "call_id": "call_a", "output": "file text"}],
                "tools": [{"type": "function", "name": "shell_command", "parameters": {"type": "object"}}]}
        prompt, tools, choice, _, mapping = convert_request(body)
        self.assertIn('"tool_use_id":"call_a"', prompt)
        output = convert_answer('{"content":[{"type":"tool_use","name":"shell_command","input":{"command":"ls"}}]}', tools, choice, mapping)
        self.assertEqual(output[0]["type"], "function_call")
        self.assertEqual(json.loads(output[0]["arguments"]), {"command": "ls"})
        events = list(output_events(output))
        self.assertEqual(events[-1]["item"], output[0])

    def test_custom_tool_and_namespace(self):
        _, tools, choice, _, mapping = convert_request({"input": "Fix it", "tools": [{"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "apply_patch"}]}]})
        output = convert_answer('{"content":[{"type":"tool_use","name":"functions.apply_patch","input":{"input":"*** Begin Patch\\n*** End Patch"}}]}', tools, choice, mapping)
        self.assertEqual(output[0]["namespace"], "functions")
        self.assertEqual(output[0]["type"], "custom_tool_call")
        self.assertIn("*** Begin Patch", output[0]["input"])

    def test_unsupported_state_rejected(self):
        with self.assertRaises(ProtocolError):
            convert_request({"input": "x", "previous_response_id": "old"})

    def test_images_reduced_to_text_markers(self):
        data = "data:image/png;base64," + "A" * 200_000
        prompt = convert_request({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": data}]}]})[0]
        self.assertIn("input_image", prompt)
        self.assertIn("omitted", prompt)
        self.assertNotIn("AAAA", prompt)

    def test_external_plugin_tools_are_not_exposed_on_notion_model(self):
        prompt, tools, _, _, mapping = convert_request({"input": "Fix a file", "tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
            {"type": "namespace", "name": "mcp__github", "tools": [{"type": "function", "name": "push", "description": "PRIVATE_PLUGIN_TOOL", "parameters": {"type": "object"}}]}]})
        self.assertEqual(list(mapping), ["exec_command"])
        self.assertNotIn("PRIVATE_PLUGIN_TOOL", prompt)


if __name__ == "__main__":
    unittest.main()
