"""MongoDB layer tests against an in-memory fake; no network is used."""

import io
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

# Set before the app loads .env, so nothing here can reach a real cluster.
os.environ["MONGODB_URI"] = ""

from PIL import Image
from pymongo.errors import PyMongoError

import db
from app import app

URI = "mongodb+srv://someone:<db_password>@cluster0.example.mongodb.net/"


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, key, direction):
        self.rows.sort(key=lambda row: row[key], reverse=direction < 0)
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def __iter__(self):
        return iter(self.rows)


class FakeCollection:
    def __init__(self, fail=False):
        self.docs = []
        self.fail = fail

    def insert_one(self, document):
        if self.fail:
            raise PyMongoError("unreachable")
        self.docs.append({"_id": len(self.docs) + 1, **document})

    def find(self):
        if self.fail:
            raise PyMongoError("unreachable")
        return FakeCursor(self.docs)


class FakeDatabase:
    def __init__(self, fail=False):
        self.collections = {}
        self.fail = fail

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeCollection(self.fail))


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(output, format="PNG")
    return output.getvalue()


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        db._warned.clear()
        db._last_ok = None
        self.fake = FakeDatabase()
        patches = [
            mock.patch.object(db, "_database", lambda: self.fake),
            # Run "background" writes inline so assertions are deterministic.
            mock.patch.object(db, "_run", lambda operation: operation()),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def env(self, **values):
        patch = mock.patch.dict(os.environ, values)
        patch.start()
        self.addCleanup(patch.stop)

    def test_nothing_is_written_when_unconfigured(self):
        self.env(MONGODB_URI="")
        db.record_removal(filename="a.png")
        self.assertFalse(db.configured())
        self.assertEqual(self.fake.collections, {})

    def test_unfilled_password_placeholder_disables_the_database(self):
        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="")
        self.assertIsNone(db.connection_uri())
        self.assertFalse(db.status()["configured"])

    def test_password_is_substituted_and_url_encoded(self):
        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="p@ss/w:rd")
        self.assertEqual(
            db.connection_uri(),
            "mongodb+srv://someone:p%40ss%2Fw%3Ard@cluster0.example.mongodb.net/",
        )

    def test_a_uri_without_the_placeholder_is_used_as_written(self):
        uri = "mongodb+srv://someone:secret@cluster0.example.mongodb.net/"
        self.env(MONGODB_URI=uri, MONGODB_PASSWORD="ignored")
        self.assertEqual(db.connection_uri(), uri)

    def test_removal_is_recorded_with_a_timestamp(self):
        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="pw")
        db.record_removal(filename="a.png", quality="best")
        (document,) = self.fake["removals"].docs
        self.assertEqual(document["filename"], "a.png")
        self.assertIsInstance(document["created_at"], datetime)
        self.assertTrue(db.status()["connected"])

    def test_a_write_failure_is_swallowed_and_reported(self):
        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="pw")
        self.fake.fail = True
        db.record_removal(filename="a.png")  # must not raise
        self.assertIs(db.status()["connected"], False)

    def test_remove_route_logs_the_job_and_still_returns_the_image(self):
        import app as app_module

        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="pw")
        with mock.patch.object(
            app_module, "create_cutout", lambda *a, **k: (png_bytes(), "balanced")
        ):
            response = app.test_client().post(
                "/remove",
                data={
                    "image": (io.BytesIO(png_bytes()), "photo.png"),
                    "quality": "auto",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        (document,) = self.fake["removals"].docs
        self.assertEqual(document["filename"], "photo.png")
        self.assertEqual(document["requested_quality"], "auto")
        self.assertEqual(document["quality"], "balanced")
        self.assertEqual(document["input_bytes"], len(png_bytes()))
        self.assertGreaterEqual(document["duration_ms"], 0)

    def test_remove_route_survives_a_database_outage(self):
        import app as app_module

        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="pw")
        self.fake.fail = True
        with mock.patch.object(
            app_module, "create_cutout", lambda *a, **k: (png_bytes(), "best")
        ):
            response = app.test_client().post(
                "/remove",
                data={"image": (io.BytesIO(png_bytes()), "photo.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)

    def test_feedback_is_saved_locally_but_not_to_mongodb(self):
        import app as app_module

        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="pw")
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            app_module, "FEEDBACK_DATA_DIR", folder
        ):
            response = app.test_client().post(
                "/feedback",
                data={
                    "image": (io.BytesIO(png_bytes()), "photo.png"),
                    "current": (io.BytesIO(png_bytes()), "current.png"),
                    "quality": "fast",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(os.listdir(folder)), 1)
        self.assertEqual(self.fake.collections, {})

    def test_history_requires_configuration(self):
        self.env(MONGODB_URI="")
        self.assertEqual(app.test_client().get("/api/history").status_code, 503)

    def test_history_returns_newest_first(self):
        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="pw")
        now = datetime.now(timezone.utc)
        for name in ("first.png", "second.png"):
            self.fake["removals"].insert_one({"filename": name, "created_at": now})
        response = app.test_client().get("/api/history?limit=1")
        self.assertEqual(response.status_code, 200)
        rows = response.get_json()["removals"]
        self.assertEqual([row["filename"] for row in rows], ["second.png"])
        self.assertNotIn("_id", rows[0])

    def test_history_rejects_a_bad_limit(self):
        self.env(MONGODB_URI=URI, MONGODB_PASSWORD="pw")
        self.assertEqual(
            app.test_client().get("/api/history?limit=abc").status_code, 400
        )

    def test_health_reports_database_state(self):
        self.env(MONGODB_URI="")
        body = app.test_client().get("/health").get_json()
        self.assertEqual(body["database"]["configured"], False)


if __name__ == "__main__":
    unittest.main()
