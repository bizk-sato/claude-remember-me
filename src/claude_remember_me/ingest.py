"""Ingest CLI: reads Stop Hook stdin, parses transcript, saves to DB.

main() は非同期モードで動作する:
  1. stdin から JSON を読み取る
  2. --worker フラグ付きで自身をバックグラウンド起動
  3. 即座に終了（hook は即座に制御を返す）

--worker モードでは実際の embedding + DB 保存を実行する。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ログファイルパス: ユーザー固有の一時ディレクトリを使用
LOG_PATH = Path(tempfile.gettempdir()) / f"claude-remember-me-{os.getuid()}.log"


def run_ingest(
    session_id: str,
    transcript_path: str,
    conn=None,
    embedder=None,
    db_path: str | None = None,
) -> int:
    """Transcript を読み込み、差分チャンクを DB に保存する。保存件数を返す。"""
    from claude_remember_me.chunker import parse_transcript
    from claude_remember_me.db import (
        DEFAULT_DB_PATH,
        get_connection,
        init_db,
        insert_memory,
        get_last_chunk_index,
        update_ingest_state,
    )
    from claude_remember_me.embedder import Embedder

    path = Path(transcript_path)
    if not path.exists():
        return 0

    jsonl_text = path.read_text(encoding="utf-8")
    pairs = parse_transcript(jsonl_text)
    if not pairs:
        return 0

    if conn is None:
        resolved_db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        conn = get_connection(resolved_db_path)
        init_db(conn)
    if embedder is None:
        embedder = Embedder()

    last_index = get_last_chunk_index(conn, session_id)
    new_pairs = [p for p in pairs if p.chunk_index > last_index]
    if not new_pairs:
        return 0

    # バッチでベクトル化（user_message + assistant_message を結合してベクトル化）
    texts = [f"{p.user_message}\n{p.assistant_message}" for p in new_pairs]
    embeddings = embedder.embed_batch(texts)

    for pair, embedding in zip(new_pairs, embeddings):
        insert_memory(
            conn,
            session_id=session_id,
            chunk_index=pair.chunk_index,
            user_message=pair.user_message,
            assistant_message=pair.assistant_message,
            project_path=pair.project_path,
            embedding=embedding,
        )

    update_ingest_state(conn, session_id, new_pairs[-1].chunk_index)
    conn.commit()  # single commit for the whole batch
    return len(new_pairs)


def _worker_main():
    """バックグラウンドワーカー: 実際の ingest 処理を実行する。"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--transcript-path", required=True)
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    # コマンドライン引数 > 環境変数 > デフォルト
    db_path = args.db_path or os.environ.get("CLAUDE_REMEMBER_ME_DB_PATH")

    saved = run_ingest(
        session_id=args.session_id,
        transcript_path=args.transcript_path,
        db_path=db_path,
    )
    if saved > 0:
        print(f"claude-remember-me: saved {saved} new memories", file=sys.stderr)


def main():
    """Stop Hook のエントリポイント。stdin から JSON を読み取り、ワーカーをバックグラウンド起動する。"""
    try:
        stdin_data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError, ValueError):
        return

    session_id = stdin_data.get("session_id")
    transcript_path = stdin_data.get("transcript_path")

    if not session_id or not transcript_path:
        return

    # バックグラウンドでワーカープロセスを起動
    cmd = [
        sys.executable, "-m", "claude_remember_me.ingest",
        "--worker",
        "--session-id", session_id,
        "--transcript-path", transcript_path,
    ]

    # --db-path が環境変数で渡された場合はワーカーにも渡す
    db_path = os.environ.get("CLAUDE_REMEMBER_ME_DB_PATH")
    if db_path:
        cmd.extend(["--db-path", db_path])

    log_file = open(LOG_PATH, "a")
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=log_file,
        start_new_session=True,
    )
    log_file.close()

    print(json.dumps({"systemMessage": "claude-remember-me: バックグラウンドで記憶を保存中..."}))


if __name__ == "__main__":
    # 第1引数が --worker の場合はワーカーモードで実行
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main()
    else:
        main()
