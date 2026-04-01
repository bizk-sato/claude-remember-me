from claude_remember_me.db import init_db, insert_memory, get_last_chunk_index, update_ingest_state


def test_init_db_creates_tables(db_conn):
    init_db(db_conn)
    cursor = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert "memories" in tables
    assert "memories_fts" in tables
    assert "memories_vec" in tables
    assert "ingest_state" in tables


def test_insert_memory(db_conn):
    init_db(db_conn)
    embedding = [0.1] * 768
    insert_memory(
        db_conn,
        session_id="sess-1",
        chunk_index=0,
        user_message="hello",
        assistant_message="hi there",
        project_path="/tmp/project",
        embedding=embedding,
    )
    db_conn.commit()
    row = db_conn.execute("SELECT * FROM memories WHERE session_id='sess-1'").fetchone()
    assert row is not None


def test_insert_duplicate_is_ignored(db_conn):
    init_db(db_conn)
    embedding = [0.1] * 768
    for _ in range(2):
        insert_memory(
            db_conn,
            session_id="sess-1",
            chunk_index=0,
            user_message="hello",
            assistant_message="hi",
            project_path=None,
            embedding=embedding,
        )
        db_conn.commit()
    count = db_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 1


def test_delete_memory_cascades_to_vec(db_conn):
    """memories 削除時に memories_vec も連動削除される"""
    init_db(db_conn)
    embedding = [0.1] * 768
    insert_memory(
        db_conn,
        session_id="sess-del",
        chunk_index=0,
        user_message="to be deleted",
        assistant_message="response",
        project_path=None,
        embedding=embedding,
    )
    db_conn.commit()
    row_id = db_conn.execute(
        "SELECT id FROM memories WHERE session_id='sess-del'"
    ).fetchone()[0]
    assert db_conn.execute(
        "SELECT COUNT(*) FROM memories_vec WHERE id=?", (row_id,)
    ).fetchone()[0] == 1

    db_conn.execute("DELETE FROM memories WHERE id=?", (row_id,))
    db_conn.commit()
    assert db_conn.execute(
        "SELECT COUNT(*) FROM memories_vec WHERE id=?", (row_id,)
    ).fetchone()[0] == 0


def test_get_last_chunk_index_returns_negative_one_for_new_session(db_conn):
    init_db(db_conn)
    assert get_last_chunk_index(db_conn, "new-session") == -1


def test_update_and_get_ingest_state(db_conn):
    init_db(db_conn)
    update_ingest_state(db_conn, "sess-1", 5)
    db_conn.commit()
    assert get_last_chunk_index(db_conn, "sess-1") == 5
    update_ingest_state(db_conn, "sess-1", 10)
    db_conn.commit()
    assert get_last_chunk_index(db_conn, "sess-1") == 10
