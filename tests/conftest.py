import pytest
import sqlite3
import sqlite_vec


@pytest.fixture
def db_conn(tmp_path):
    """テスト用の一時 SQLite DB を返す"""
    db_path = tmp_path / "test_memory.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    yield conn
    conn.close()


@pytest.fixture
def populated_db(db_conn):
    """テストデータ入りの DB を返す"""
    from claude_remember_me.db import init_db, insert_memory

    init_db(db_conn)
    test_data = [
        ("s1", 0, "Pythonの型ヒントについて教えて", "Pythonの型ヒントはmypyで検証できます", "/project-a"),
        ("s1", 1, "Rustのライフタイムとは？", "Rustのライフタイムはメモリ安全性を保証する仕組みです", "/project-a"),
        ("s2", 0, "SQLiteのFTS5の使い方", "FTS5はSQLiteの全文検索拡張です", "/project-b"),
        ("s2", 1, "今日のランチは何にする？", "カレーがおすすめです", "/project-b"),
    ]
    for sid, idx, user_msg, asst_msg, path in test_data:
        embedding = [0.0] * 768
        for i, ch in enumerate(user_msg[:10]):
            embedding[i % 768] = ord(ch) / 10000.0
        insert_memory(
            db_conn,
            session_id=sid,
            chunk_index=idx,
            user_message=user_msg,
            assistant_message=asst_msg,
            project_path=path,
            embedding=embedding,
        )
    db_conn.commit()
    return db_conn
