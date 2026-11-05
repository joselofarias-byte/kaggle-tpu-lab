#!/usr/bin/env python3
"""Generate an OpenCode provider without writing the API key to disk."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit


def provider(url, model, context, output):
    parsed = urlsplit(url)
    local = parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}
    if not (parsed.scheme == 'https' or local) or not parsed.hostname:
        raise ValueError('HTTPS required except on loopback')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Credentials, query strings and fragments are forbidden in endpoint URLs')
    if not model.strip() or not 0 < output <= context:
        raise ValueError('Model and valid measured context/output limits required')
    base = url.rstrip('/').removesuffix('/v1') + '/v1'
    return {'npm': '@ai-sdk/openai-compatible', 'name': 'Authorized LLM lab',
            'options': {'baseURL': base, 'apiKey': '{env:MOLAB_API_KEY}'},
            'models': {model: {'name': model, 'limit': {'context': context, 'output': output}}}}


def write_config(path, entry):
    path = Path(path).expanduser()
    data = json.loads(path.read_text()) if path.exists() else {}
    providers = data.setdefault('provider', {})
    if 'jolufasoft-lab' in providers:
        raise ValueError('Provider jolufasoft-lab already exists; choose another config file or edit explicitly')
    providers['jolufasoft-lab'] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.opencode-', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Authorized endpoint; never a Molab tunnel')
    parser.add_argument('--model', required=True)
    parser.add_argument('--context', type=int, required=True, help='Context actually configured on server')
    parser.add_argument('--output', type=int, default=2048)
    parser.add_argument('--config', default='~/.config/opencode/opencode.json')
    args = parser.parse_args()
    try:
        write_config(args.config, provider(args.url, args.model, args.context, args.output))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Configuration refused: {exc}\n')
    print('Configured provider jolufasoft-lab. Set MOLAB_API_KEY privately in the client environment.')
    print('Tool support depends on the model, template and backend; run the API smoke test first.')


if __name__ == '__main__':
    main()
