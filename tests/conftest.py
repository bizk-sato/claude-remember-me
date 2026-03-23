import pytest
import sqlite3
import sqlite_vec


@pytest.fixture
def db_conn(tmp_path):
    """テスト用の一時 SQLite DB を返す"""
    db_path = tmp_path / "test_memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    yield conn
    conn.close()
