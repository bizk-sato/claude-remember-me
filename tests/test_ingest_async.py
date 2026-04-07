"""非同期 ingest のテスト

Stop Hook から呼ばれる main() が即座に返り、
重い embedding 処理はバックグラウンドで実行されることを検証する。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _make_transcript(tmp_path, pairs):
    """テスト用の transcript JSONL を作成"""
    transcript_path = tmp_path / "transcript.jsonl"
    lines = []
    for user_msg, asst_msg in pairs:
        lines.append(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": user_msg},
            "sessionId": "test-async-session",
            "cwd": "/tmp/project",
            "userType": "external",
        }))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": asst_msg}]},
        }))
    transcript_path.write_text("\n".join(lines))
    return str(transcript_path)


class TestIngestAsyncMain:
    """main() が非同期モードで即座に返ることを検証"""

    def test_main_returns_within_2_seconds(self, tmp_path):
        """main() は重い処理をバックグラウンドに委譲し、2秒以内に返る"""
        transcript_path = _make_transcript(tmp_path, [
            ("question 1", "answer 1"),
            ("question 2", "answer 2"),
        ])

        stdin_data = json.dumps({
            "session_id": "test-async-session",
            "transcript_path": transcript_path,
        })

        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", "claude_remember_me.ingest"],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=10,
        )
        elapsed = time.monotonic() - start

        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        assert elapsed < 2.0, f"main() took {elapsed:.1f}s — should return within 2s"
        output = json.loads(proc.stdout)
        assert "バックグラウンドで記憶を保存中" in output["systemMessage"]

    def test_main_with_empty_stdin_returns_immediately(self):
        """空のstdinでもエラーなく即座に返る"""
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", "claude_remember_me.ingest"],
            input="",
            capture_output=True,
            text=True,
            timeout=5,
        )
        elapsed = time.monotonic() - start

        assert proc.returncode == 0
        assert elapsed < 2.0


class TestBackgroundIngestCompletes:
    """バックグラウンドプロセスが実際に ingest を完了することを検証"""

    def test_background_process_saves_to_db(self, tmp_path, db_conn):
        """main() 呼び出し後、バックグラウンドで DB にメモリが保存される"""
        from claude_remember_me.db import init_db
        init_db(db_conn)

        transcript_path = _make_transcript(tmp_path, [
            ("async question 1", "async answer 1"),
            ("async question 2", "async answer 2"),
        ])

        db_path = str(tmp_path / "test_memory.db")

        stdin_data = json.dumps({
            "session_id": "test-async-session",
            "transcript_path": transcript_path,
        })

        # main() を呼び出し（バックグラウンドプロセスを起動）
        env = {**os.environ, "CLAUDE_REMEMBER_ME_DB_PATH": db_path}
        proc = subprocess.run(
            [sys.executable, "-m", "claude_remember_me.ingest"],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}"

        # バックグラウンドプロセスの完了を待つ（最大60秒）
        deadline = time.monotonic() + 60
        saved_count = 0
        while time.monotonic() < deadline:
            from claude_remember_me.db import get_connection
            check_conn = get_connection(Path(db_path))
            init_db(check_conn)
            saved_count = check_conn.execute(
                "SELECT COUNT(*) FROM memories WHERE session_id = ?",
                ("test-async-session",),
            ).fetchone()[0]
            check_conn.close()
            if saved_count >= 2:
                break
            time.sleep(1)

        assert saved_count == 2, (
            f"Expected 2 memories saved by background process, got {saved_count}"
        )
