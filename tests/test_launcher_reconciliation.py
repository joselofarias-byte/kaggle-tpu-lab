import unittest
from unittest.mock import patch

import launch


class EndpointWaitTests(unittest.TestCase):
    def wait(self, status, events, **kwargs):
        with patch.object(launch, '_kernel_status', return_value=(status, status)), \
             patch.object(launch, 'read_events', return_value=events), \
             patch.object(launch.time, 'time', return_value=10000), \
             patch.object(launch.time, 'sleep', side_effect=AssertionError('unexpected wait')), \
             patch.object(launch, 'say'):
            return launch._wait_for_public_endpoint('user/kernel', 'session-topic', **kwargs)

    def test_queued_ready_is_adopted(self):
        ready = {'phase': 'ready', 'endpoint': 'https://example.com'}
        self.assertEqual(self.wait('QUEUED', [(9990, ready)]), (ready, None))

    def test_stale_terminal_status_does_not_hide_recent_ready(self):
        ready = {'phase': 'ready', 'endpoint': 'https://example.com'}
        self.assertEqual(self.wait('COMPLETE', [(9990, ready)]), (ready, None))

    def test_session_failure_wins_over_older_ready(self):
        events = [(9980, {'phase': 'ready', 'endpoint': 'https://example.com'}),
                  (9990, {'phase': 'failed'})]
        self.assertEqual(self.wait('QUEUED', events), (None, 'event-failed'))

    def test_old_ready_is_not_adopted(self):
        events = [(8000, {'phase': 'ready', 'endpoint': 'https://example.com'})]
        self.assertEqual(self.wait('COMPLETE', events), (None, 'kernel-complete'))

    def test_recent_heartbeat_refreshes_ready_evidence(self):
        ready = {'phase': 'ready', 'endpoint': 'https://example.com'}
        self.assertEqual(self.wait('QUEUED', [(8000, ready), (9990, {'phase': 'heartbeat'})]),
                         (ready, None))

    def test_tunnel_url_alone_is_not_ready(self):
        self.assertEqual(self.wait('COMPLETE', [(9990, {'phase': 'tunnel-url',
                                                       'endpoint': 'https://example.com'})]),
                         (None, 'kernel-complete'))

    def test_boot_events_start_timeout_even_when_slug_stays_queued(self):
        self.assertEqual(self.wait('QUEUED', [(9990, {'phase': 'server-launch'})],
                                   start_timeout_s=0), (None, 'startup-timeout'))

    def test_genuine_queue_can_timeout_without_boot(self):
        with patch.object(launch, '_kernel_status', return_value=('QUEUED', '')), \
             patch.object(launch, 'read_events', return_value=[]), \
             patch.object(launch.time, 'time', side_effect=[10000, 10002]), \
             patch.object(launch, 'say'):
            self.assertEqual(launch._wait_for_public_endpoint('u/k', 'topic', queue_timeout_s=1),
                             (None, 'queue-timeout'))


class AutoAuditAmbiguousSessionTests(unittest.TestCase):
    def test_pending_or_unknown_status_never_starts_another_tpu(self):
        from argparse import Namespace
        import tempfile
        from pathlib import Path
        for status in ("UNKNOWN", "CANCEL_ACKNOWLEDGED", "CANCELACKNOWLEDGED",
                       "CANCEL_REQUESTED", "CANCELREQUESTED"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as repo, \
                 patch.object(launch, '_install_qwen_audit_shortcut'), \
                 patch.object(launch, '_pick_audit_instructions', return_value=Path("prompt.md")), \
                 patch.object(launch, 'check_auth'), \
                 patch.object(launch, 'STATE_FILE') as state_file, \
                 patch.object(launch, 'load_state', return_value={"kernel": "u/k", "topic": "session"}), \
                 patch.object(launch, '_kernel_status', return_value=(status, "")), \
                 patch.object(launch.subprocess, 'Popen') as start, \
                 patch.object(launch, 'cmd_stop') as stop:
                state_file.exists.return_value = True
                with self.assertRaisesRegex(SystemExit, "No voy a iniciar otra TPU"):
                    launch.cmd_auto_audit(Namespace(repo=repo, instructions=None))
                start.assert_not_called()
                stop.assert_not_called()

if __name__ == '__main__':
    unittest.main()
