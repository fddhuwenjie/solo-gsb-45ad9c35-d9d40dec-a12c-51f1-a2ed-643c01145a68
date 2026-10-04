"""Batch certificate endpoint tests.

Creation reuses the release verification: only a RELEASED verdict is
certified, freezing layout/scan/readings/thresholds and the blinded-view
digest into one SQLite transaction. Re-verification restores the frozen
parameters; success appends a new immutable evidence revision (the fresh
scan/readings get their own digest) and failure revokes the certificate on
any misplacement, QC failure or view-digest change (first failure reason
returned) without producing a new revision. A revoked certificate can never
be restored and stale revisions can never overwrite. Evidence revisions and
idempotency records survive a process restart. Idempotency: the same
request_key with identical content replays the recorded response; different
content is 409.
"""
import hashlib
import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app import certificates
from app.main import app
from app.schemas import CertificateReverifyRequest

client = TestClient(app)

# 4 rows x 4 cols, 16 wells, two groups interleaved checkerboard-style;
# integer noise gives small, well-converged row/column effects.
LAYOUT_4X4 = {
    "gA1": "A1", "gA2": "A3", "gB1": "A2", "gB2": "A4",
    "gB3": "B1", "gB4": "B3", "gA3": "B2", "gA4": "B4",
    "gA5": "C1", "gA6": "C3", "gB5": "C2", "gB6": "C4",
    "gB7": "D1", "gB8": "D3", "gA7": "D2", "gA8": "D4",
}
SAMPLES_4X4 = [
    {"id": s, "group": "GA" if s.startswith("gA") else "GB"}
    for s in LAYOUT_4X4
]
PLATE_4X4 = {"type": "custom", "rows": 4, "columns": 4}


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    """Isolate every test in its own SQLite database."""
    monkeypatch.setenv("CERT_DB_PATH", str(tmp_path / "certificates.db"))
    certificates.reset_stores()
    yield tmp_path / "certificates.db"
    certificates.reset_stores()


def _hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:6], 16)


def _readings(row_bump=0.0, offset=0.0):
    out = {}
    for sid, well in LAYOUT_4X4.items():
        base = 100.0 if sid.startswith("gA") else 0.0
        noise = (_hash(sid) % 7) - 3  # -3..3
        out[well] = (
            base
            + (row_bump if well[0] == "A" else 0.0)
            + noise
            + offset
        )
    return out


def _release_payload(**over):
    payload = {
        "plate": PLATE_4X4,
        "target": dict(LAYOUT_4X4),
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "blocked_wells": [],
        "samples": [dict(s) for s in SAMPLES_4X4],
        "readings": _readings(),
        "seed": 7,
        "max_spatial_effect": 10.0,
        "min_p_value": 0.001,
        "permutations": 100,
        "tolerance": 1e-6,
        "max_iterations": 100,
    }
    payload.update(over)
    return payload


def _view():
    """Blinded operational view over the layout wells (row-major order)."""
    wells = sorted(LAYOUT_4X4.values(), key=lambda w: (w[0], int(w[1:])))
    return [{"well": w, "blind_id": f"BL-{i:02d}"} for i, w in enumerate(wells)]


def _create_payload(**over):
    payload = {
        "batch_id": "BATCH-1",
        "request_key": "create-1",
        "release": _release_payload(),
        "view": _view(),
    }
    payload.update(over)
    return payload


def _reverify_payload(**over):
    payload = {
        "batch_id": "BATCH-1",
        "request_key": "reverify-1",
        "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        # Fresh readings that still pass every gate (a constant offset does
        # not change spatial effects); distinct from the creation evidence.
        "readings": _readings(offset=0.5),
        "view": _view(),
    }
    payload.update(over)
    return payload


def _create_cert(**over):
    r = client.post("/api/v1/certificates", json=_create_payload(**over))
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------- create


def test_create_success_returns_certificate_fields():
    r = client.post("/api/v1/certificates", json=_create_payload())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["certificate_id"].startswith("cert-")
    assert body["batch_id"] == "BATCH-1"
    assert body["status"] == "ACTIVE"
    assert body["revision"] == 1
    assert len(body["digest"]) == 64 and int(body["digest"], 16) >= 0
    assert len(body["view_digest"]) == 64


def test_create_without_view_has_null_view_digest():
    payload = _create_payload()
    del payload["view"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "ACTIVE"
    assert body["view_digest"] is None


def test_create_not_released_creates_no_certificate():
    payload = _create_payload()
    payload["release"]["max_spatial_effect"] = 0.01  # effect gate fails
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["certificate_id"] is None
    assert body["status"] == "NOT_CERTIFIED"
    assert body["verdict"] == "NOT_RELEASED"
    assert body["reasons"] == ["SPATIAL_EFFECT_EXCEEDED"]
    # Nothing was persisted: re-verification finds no certificate.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"


def test_create_scan_mismatch_is_422_and_creates_nothing():
    payload = _create_payload()
    del payload["release"]["scan"]["A1"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert "SAMPLES_MISSING" in body["infeasibility_categories"]
    assert body["misplaced_wells"] == [
        {"well": "A1", "expected": "gA1", "actual": None}]
    assert "certificate_id" not in body
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404


def test_create_replay_same_key_same_content_returns_recorded_response():
    payload = _create_payload()
    b1 = client.post("/api/v1/certificates", json=payload)
    b2 = client.post("/api/v1/certificates", json=payload)
    assert b1.status_code == 201 and b2.status_code == 201
    assert b1.json() == b2.json()


def test_create_same_key_different_content_is_409():
    _create_cert()
    payload = _create_payload()
    payload["release"]["seed"] = 8  # same request_key, different content
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "REQUEST_KEY_CONFLICT"
    assert body["request_key"] == "create-1"


def test_create_second_certificate_for_same_batch_is_409():
    first = _create_cert()
    r = client.post("/api/v1/certificates",
                    json=_create_payload(request_key="create-2"))
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_ALREADY_EXISTS"
    assert body["certificate_id"] == first["certificate_id"]
    assert body["status"] == "ACTIVE"


def test_create_invalid_view_is_422_and_creates_nothing():
    payload = _create_payload()
    payload["view"][1]["blind_id"] = payload["view"][0]["blind_id"]  # duplicate
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert "CERTIFICATE_VIEW_INVALID" in r.json()["infeasibility_categories"]

    payload = _create_payload(request_key="create-2")
    payload["view"][0]["well"] = "Z9"  # outside the plate
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]

    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404


def test_create_not_certified_response_is_recorded_under_the_key():
    payload = _create_payload()
    payload["release"]["max_spatial_effect"] = 0.01
    b1 = client.post("/api/v1/certificates", json=payload)
    b2 = client.post("/api/v1/certificates", json=payload)
    assert b1.status_code == 200 and b2.status_code == 200
    assert b1.json() == b2.json()
    # Same key, different content -> 409; fresh key -> the batch can certify.
    payload["release"]["seed"] = 8
    assert client.post("/api/v1/certificates", json=payload).status_code == 409
    r = client.post("/api/v1/certificates",
                    json=_create_payload(request_key="create-2"))
    assert r.status_code == 201


# ------------------------------------------------------------------- reverify


def test_reverify_success_appends_immutable_evidence_revision(fresh_store):
    created = _create_cert()
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ACTIVE"
    assert body["revision"] == 2
    assert body["certificate_id"] == created["certificate_id"]
    # The digest is recomputed over the new evidence snapshot — it covers the
    # fresh readings and therefore differs from the creation digest.
    assert body["digest"] != created["digest"]
    assert body["view_digest"] == created["view_digest"]

    db = sqlite3.connect(str(fresh_store))
    rev_rows = db.execute(
        "SELECT revision, digest, snapshot FROM certificate_revisions "
        "ORDER BY revision"
    ).fetchall()
    assert [row[0] for row in rev_rows] == [1, 2]
    rev1, rev2 = rev_rows
    # Each revision's digest equals SHA-256 over its own stored snapshot.
    assert rev1[1] == created["digest"]
    snap1, snap2 = json.loads(rev1[2]), json.loads(rev2[2])
    assert snap1["readings"] == _readings()
    assert snap2["readings"] == _readings(offset=0.5)
    canonical = json.dumps(
        snap2, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == body["digest"] == rev2[1]
    # Revision 1 evidence is byte-identical to the creation snapshot, while
    # the certificate row now points at the revision-2 evidence.
    cert_row = db.execute(
        "SELECT digest, snapshot, revision FROM certificates").fetchone()
    assert cert_row[2] == 2 and cert_row[0] == body["digest"]
    assert cert_row[1] == rev2[2]
    assert rev1[2] != rev2[2]
    db.close()

    # A further successful re-verification appends revision 3 evidence.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2", expected_revision=2,
        readings=_readings(offset=1.5)))
    assert r.status_code == 200
    third = r.json()
    assert third["revision"] == 3
    assert third["digest"] not in {created["digest"], body["digest"]}


def test_reverify_replay_does_not_double_increment():
    _create_cert()
    payload = _reverify_payload()
    b1 = client.post("/api/v1/certificates/reverify", json=payload)
    b2 = client.post("/api/v1/certificates/reverify", json=payload)
    assert b1.status_code == 200 and b2.status_code == 200
    assert b1.json() == b2.json()
    assert b2.json()["revision"] == 2
    # The replay did not burn a revision: 2 is still the current one.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2", expected_revision=2))
    assert r.status_code == 200
    assert r.json()["revision"] == 3


def test_reverify_same_key_different_content_is_409():
    _create_cert()
    client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    payload = _reverify_payload(expected_revision=2)
    payload["readings"] = _readings(row_bump=0.5)  # same key, new content
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"


def test_reverify_stale_revision_is_409_and_never_overwrites():
    _create_cert()
    client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    # Current revision is 2; resubmitting against the old revision 1 fails.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2", expected_revision=1))
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert body["current_revision"] == 2
    assert body["expected_revision"] == 1
    # The certificate is untouched: the correct revision still verifies.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-3", expected_revision=2))
    assert r.status_code == 200
    assert r.json()["revision"] == 3


def test_reverify_replay_returns_new_evidence_without_extra_revision(fresh_store):
    created = _create_cert()
    payload = _reverify_payload()
    b1 = client.post("/api/v1/certificates/reverify", json=payload)
    b2 = client.post("/api/v1/certificates/reverify", json=payload)
    assert b1.status_code == 200 and b2.status_code == 200
    assert b1.json() == b2.json()
    assert b2.json()["revision"] == 2
    assert b2.json()["digest"] != created["digest"]
    db = sqlite3.connect(str(fresh_store))
    # Replay must not append a second revision-2 row or any revision 3.
    rows = db.execute(
        "SELECT revision, COUNT(*) FROM certificate_revisions GROUP BY revision"
    ).fetchall()
    assert rows == [(1, 1), (2, 1)]
    db.close()


def test_evidence_revisions_and_replay_survive_restart(fresh_store):
    created = _create_cert()
    first = client.post(
        "/api/v1/certificates/reverify", json=_reverify_payload()).json()
    second = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2", expected_revision=2,
        readings=_readings(offset=1.5))).json()
    assert second["revision"] == 3

    # Simulate a process restart: drop every cached connection and reopen.
    certificates.reset_stores()
    store = certificates.get_store()
    try:
        status, body = certificates.reverify_certificate(
            CertificateReverifyRequest(**_reverify_payload()))
        assert status == 200
        # The idempotency record survived the restart: identical key+content
        # replays revision 2 and never appends a new revision.
        assert body == first
        assert body["revision"] == 2 and body["digest"] == first["digest"]

        with store.transaction() as conn:
            cert = certificates._get_certificate(conn, "BATCH-1")
            assert cert["status"] == "ACTIVE"
            assert cert["revision"] == 3
            # The current snapshot/digest are the revision-3 evidence...
            assert cert["digest"] == second["digest"]
            rows = conn.execute(
                "SELECT revision, digest, snapshot FROM certificate_revisions "
                "ORDER BY revision").fetchall()
            assert [row["revision"] for row in rows] == [1, 2, 3]
            # ...and every historical revision is still auditable verbatim.
            assert rows[0]["digest"] == created["digest"]
            assert rows[1]["digest"] == first["digest"]
            assert rows[2]["digest"] == second["digest"]
            assert json.loads(rows[0]["snapshot"])["readings"] == _readings()
            assert json.loads(rows[1]["snapshot"])["readings"] == \
                _readings(offset=0.5)
            assert json.loads(rows[2]["snapshot"])["readings"] == \
                _readings(offset=1.5)
            for row in rows:
                assert certificates._sha256(
                    row["snapshot"].encode("utf-8")) == row["digest"]
    finally:
        certificates.reset_stores()


def test_failed_reverify_appends_no_revision_and_keeps_old_evidence(fresh_store):
    created = _create_cert()
    payload = _reverify_payload(readings=_readings(row_bump=100.0))
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revision"] == 1
    assert body["digest"] == created["digest"]
    db = sqlite3.connect(str(fresh_store))
    # No new evidence revision was produced; revision 1 is untouched.
    rows = db.execute(
        "SELECT revision, digest FROM certificate_revisions ORDER BY revision"
    ).fetchall()
    assert rows == [(1, created["digest"])]
    cert_row = db.execute(
        "SELECT status, revision, digest FROM certificates").fetchone()
    assert cert_row == ("REVOKED", 1, created["digest"])
    db.close()


def test_concurrent_reverify_serialises_to_one_revision(fresh_store):
    _create_cert()

    def request(expected_revision, key):
        return certificates.reverify_certificate(
            CertificateReverifyRequest(**_reverify_payload(
                request_key=key, expected_revision=expected_revision)))

    # Two threads racing at revision 1: exactly one may append revision 2;
    # the other loses the optimistic update guard as a stale revision.
    outcomes = {}

    def worker(name, key, barrier):
        barrier.wait()
        outcomes[name] = request(1, key)

    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(target=worker, args=("a", "rv-a", barrier)),
        threading.Thread(target=worker, args=("b", "rv-b", barrier)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    statuses = sorted(status for status, _ in outcomes.values())
    assert statuses == [200, 409]
    winner = next(body for status, body in outcomes.values() if status == 200)
    loser = next(body for status, body in outcomes.values() if status == 409)
    assert winner["revision"] == 2
    assert loser["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert loser["current_revision"] == 2 and loser["expected_revision"] == 1

    db = sqlite3.connect(str(fresh_store))
    rows = db.execute(
        "SELECT revision, COUNT(*) FROM certificate_revisions GROUP BY revision"
    ).fetchall()
    assert rows == [(1, 1), (2, 1)]
    db.close()

    # The stale loser's key was not recorded (state-dependent 409s never
    # burn a key); retrying it against the new revision succeeds and appends
    # revision 3. The winner's recorded key cannot carry new content.
    winner_key = "rv-a" if outcomes["a"][0] == 200 else "rv-b"
    loser_key = "rv-b" if winner_key == "rv-a" else "rv-a"
    status, body = request(2, loser_key)
    assert status == 200 and body["revision"] == 3
    status, body = request(3, winner_key)
    assert status == 409 and body["error"] == "REQUEST_KEY_CONFLICT"


def test_reverify_unknown_batch_is_404():
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"


def test_reverify_closed_loop_misplacement_revokes_with_first_reason():
    created = _create_cert()
    payload = _reverify_payload()
    payload["scan"]["A1"], payload["scan"]["A2"] = "gB1", "gA1"  # swap
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revision"] == 1  # never incremented
    assert body["digest"] == created["digest"]
    revocation = body["revocation"]
    assert revocation["reason"] == "CLOSED_LOOP_MISPLACEMENT"
    assert revocation["cycles"] == [["A1", "A2"]]
    assert {"well": "A1", "expected": "gA1", "actual": "gB1"} in \
        revocation["misplaced_wells"]


def test_reverify_missing_sample_revokes():
    _create_cert()
    payload = _reverify_payload()
    del payload["scan"]["A1"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revocation"]["reason"] == "SAMPLES_MISSING"


def test_reverify_qc_gate_failure_revokes():
    _create_cert()
    # Fresh readings carry a strong row-A effect that breaches the frozen
    # max_spatial_effect=10.0 threshold.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        readings=_readings(row_bump=100.0)))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revocation"]["reason"] == "SPATIAL_EFFECT_EXCEEDED"


def test_reverify_structural_reading_problem_revokes():
    _create_cert()
    payload = _reverify_payload()
    payload["readings"]["a1"] = 1.0  # duplicates "A1" after normalisation
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "REVOKED"
    assert r.json()["revocation"]["reason"] == "SPATIAL_READING_DUPLICATED"


def test_reverify_view_digest_change_revokes():
    _create_cert()
    payload = _reverify_payload()
    payload["view"][0]["blind_id"] = "BL-tampered"
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revocation"]["reason"] == "VIEW_DIGEST_MISMATCH"


def test_reverify_with_matching_view_passes():
    _create_cert()
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 200
    assert r.json()["status"] == "ACTIVE"


def test_reverify_without_view_passes():
    _create_cert()  # certificate carries a view digest
    payload = _reverify_payload()
    del payload["view"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "ACTIVE"


def test_reverify_invalid_view_is_422_and_leaves_certificate_active():
    _create_cert()
    payload = _reverify_payload()
    payload["view"][0]["well"] = "Z9"
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]
    # Not a verification failure: the certificate is still ACTIVE at rev 1.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2"))
    assert r.status_code == 200
    assert r.json()["revision"] == 2


def test_revoked_certificate_can_never_be_restored():
    _create_cert()
    payload = _reverify_payload()
    del payload["scan"]["A1"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.json()["status"] == "REVOKED"

    # Replaying the revoking request returns the recorded REVOKED response.
    replay = client.post("/api/v1/certificates/reverify", json=payload)
    assert replay.status_code == 200
    assert replay.json() == r.json()

    # Any new re-verification is refused; revocation is final.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2"))
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVOKED"
    assert body["revoked_reason"] == "SAMPLES_MISSING"

    # And no new certificate can be created for the batch either.
    r = client.post("/api/v1/certificates",
                    json=_create_payload(request_key="create-2"))
    assert r.status_code == 409
    assert r.json()["error"] == "CERTIFICATE_ALREADY_EXISTS"
    assert r.json()["status"] == "REVOKED"


# ------------------------------------------------------------- confidentiality


def test_certificate_stores_no_view_entries_or_plaintext_blind_mapping(fresh_store):
    _create_cert()
    conn = sqlite3.connect(str(fresh_store))
    row = conn.execute(
        "SELECT snapshot, view_digest, digest FROM certificates").fetchone()
    snapshot, view_digest, digest = row
    stored = json.loads(snapshot)
    # The snapshot freezes layout/scan/readings/thresholds plus the view
    # DIGEST only — never the view entries (well -> blind_id) themselves.
    assert stored["view_digest"] == view_digest
    assert "view" not in stored
    assert set(stored) == {
        "batch_id", "plate", "target", "scan", "readings", "samples",
        "blocked_wells", "seed", "thresholds", "view_digest",
    }
    assert len(view_digest) == 64 and len(digest) == 64
    # The immutable revision history stores the same evidence shape and never
    # any view entries either.
    rev = conn.execute(
        "SELECT snapshot, digest FROM certificate_revisions "
        "WHERE batch_id = 'BATCH-1' AND revision = 1").fetchone()
    assert rev[0] == snapshot and rev[1] == digest
    conn.close()
    # No blind_id appears anywhere in the database files (main db or WAL).
    for suffix in ("", "-wal", "-shm"):
        path = fresh_store.with_name(fresh_store.name + suffix)
        if path.exists():
            assert b"BL-" not in path.read_bytes()


# ------------------------------------------------------------ request shape


def test_missing_request_key_is_request_validation_error():
    payload = _create_payload()
    del payload["request_key"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_reverify_missing_expected_revision_is_request_validation_error():
    payload = _reverify_payload()
    del payload["expected_revision"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_unknown_field_is_422():
    payload = _create_payload()
    payload["mapping_package"] = "secret"  # not part of the certificate contract
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_existing_endpoints_remain_available():
    r = client.post("/api/v1/arrangements/batch-release", json=_release_payload())
    assert r.status_code == 200
    assert r.json()["verdict"] == "RELEASED"
    assert client.get("/health").json() == {"status": "ok"}
