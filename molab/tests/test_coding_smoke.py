import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

spec = importlib.util.spec_from_file_location('smoke', Path(__file__).parents[1] / 'scripts/smoke_test_coding_api.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
KEY = 'test-key-' + 'x' * 32


class Response(io.BytesIO):
    def __init__(self, data, content_type='application/json'):
        super().__init__(data)
        self.headers = {'Content-Type': content_type}


class FakeAPI:
    def __init__(self, done=True, function='add'):
        self.done = done
        self.function = function
        self.calls = 0

    def open(self, req, timeout):
        if req.get_header('Authorization') != 'Bearer ' + KEY:
            raise urllib.error.HTTPError(req.full_url, 401, 'Unauthorized', {}, None)
        if req.full_url.endswith('/models'):
            return Response(b'{"data":[{"id":"exact-model"}]}')
        data = json.loads(req.data)
        if data['stream']:
            chunks = [{'choices': [{'delta': {'content': 'OK'}, 'finish_reason': None}]}, {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}]
            raw = b''.join(b'data: ' + json.dumps(chunk).encode() + b'\n\n' for chunk in chunks)
            return Response(raw + (b'data: [DONE]\n\n' if self.done else b''), 'text/event-stream')
        self.calls += 1
        if self.calls == 1:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'test-call', 'type': 'function', 'function': {'name': self.function, 'arguments': '{"a":17,"b":25}'}}]}
        else:
            assert data['messages'][-1]['role'] == 'tool'
            assert data['messages'][-1]['content'] == '42'
            message = {'role': 'assistant', 'content': '42'}
        return Response(json.dumps({'choices': [{'message': message}]}).encode())


class SmokeTests(unittest.TestCase):
    def test_all_gates(self):
        with patch.object(smoke.urllib.request, 'build_opener', return_value=FakeAPI()):
            smoke.run('http://127.0.0.1:18080/v1', KEY, 'exact-model')

    def test_incomplete_sse_fails(self):
        with patch.object(smoke.urllib.request, 'build_opener', return_value=FakeAPI(done=False)):
            with self.assertRaisesRegex(RuntimeError, 'DONE'):
                smoke.run('http://127.0.0.1:18080', KEY, 'exact-model')

    def test_unapproved_tool_rejected(self):
        with patch.object(smoke.urllib.request, 'build_opener', return_value=FakeAPI(function='shell')):
            with self.assertRaisesRegex(RuntimeError, 'allowlisted'):
                smoke.run('http://127.0.0.1:18080', KEY, 'exact-model')

    def test_redirect_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'Redirect'):
            smoke.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example')
