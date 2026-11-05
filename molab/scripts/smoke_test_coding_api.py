#!/usr/bin/env python3
"""Verify an authorized API, including SSE completion and an allowlisted tool round trip.

Set MOLAB_URL, MOLAB_API_KEY and MODEL_ID. Molab allows loopback testing only.
"""
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Redirect refused to protect the API key')


def run(base, key, model):
    parsed = urlsplit(base)
    local = parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}
    if not (parsed.scheme == 'https' or local) or not parsed.hostname:
        raise ValueError('HTTPS required except on loopback')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Endpoint must not contain credentials, query or fragment')
    if len(key) < 24:
        raise ValueError('API key must contain at least 24 characters')
    base = base.rstrip('/').removesuffix('/v1') + '/v1'
    opener = urllib.request.build_opener(NoRedirect())

    def request(path, payload=None, credential=key):
        headers = {'Content-Type': 'application/json'}
        if credential is not None:
            headers['Authorization'] = 'Bearer ' + credential
        req = urllib.request.Request(base + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
        return opener.open(req, timeout=120)

    for credential in (None, key + '-wrong'):
        try:
            with request('/models', credential=credential):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise RuntimeError('Expected HTTP 401 for invalid API key') from None
        else:
            raise RuntimeError('Server accepted a missing or invalid key')
    print('PASS authentication')
    with request('/models') as response:
        models = json.load(response)
    if model not in [item['id'] for item in models['data']]:
        raise RuntimeError('Selected model not listed')
    print('PASS models')
    with request('/chat/completions', {'model': model, 'messages': [{'role': 'user', 'content': 'Respond only OK'}], 'max_tokens': 64, 'stream': True}) as response:
        if 'text/event-stream' not in response.headers.get('Content-Type', ''):
            raise RuntimeError('Not SSE')
        content = False
        done = False
        finished = False
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            data = raw[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            chunk = json.loads(data)
            if 'error' in chunk:
                raise RuntimeError('SSE returned an error')
            for choice in chunk.get('choices', []):
                content |= bool(choice.get('delta', {}).get('content'))
                finished |= choice.get('finish_reason') is not None
        if not (content and done and finished):
            raise RuntimeError('SSE missing content, finish_reason or DONE')
    print('PASS streaming')
    tools = [{'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'additionalProperties': False}}}]
    messages = [{'role': 'user', 'content': 'Use add to add 17 and 25. Do not answer directly.'}]
    payload = {'model': model, 'temperature': 0, 'max_tokens': 256, 'stream': False, 'messages': messages, 'tools': tools, 'tool_choice': 'required'}
    with request('/chat/completions', payload) as response:
        message = json.load(response)['choices'][0]['message']
    calls = message.get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != 'add':
        raise RuntimeError('Expected exactly one allowlisted add call')
    call = calls[0]
    args = json.loads(call['function']['arguments'])
    if set(args) != {'a', 'b'} or any(type(value) is not int for value in args.values()) or args != {'a': 17, 'b': 25}:
        raise RuntimeError('Invalid add arguments')
    result = args['a'] + args['b']
    messages.extend([message, {'role': 'tool', 'tool_call_id': call['id'], 'content': str(result)}])
    payload['tool_choice'] = 'none'
    with request('/chat/completions', payload) as response:
        followup = json.load(response)['choices'][0]['message']
    if '42' not in (followup.get('content') or '') or followup.get('tool_calls'):
        raise RuntimeError('Tool result was not consumed in the final answer')
    print('PASS tool calling and executed add round trip')
    print('READY: authentication, models, SSE and tool execution gates passed')


if __name__ == '__main__':
    run(os.environ['MOLAB_URL'], os.environ['MOLAB_API_KEY'], os.environ['MODEL_ID'])
