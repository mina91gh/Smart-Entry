"""
Database models for the Smart Entry face-recognition service.

Tables:
    persons          - one row per known person
    face_encodings   - one row per face embedding (pgvector type)

Run this file directly to create the tables:
    python model.py
"""

from datetime import datetime

from peewee import *
from pgvector.peewee import VectorField


# =========================
# Database
# =========================

# PostgreSQL database names are case-sensitive.
# The database was created as lowercase "face-reco" (not "face-Recog").

db = PostgresqlDatabase(
    "face-reco",
    user="postgres",
    password="123456789",
    host="localhost",
    port=5432,
)


class BaseModel(Model):

    class Meta:
        database = db


# =========================
# Models
# =========================

class Person(BaseModel):
    """A person registered in the system."""

    name = CharField(max_length=100, unique=True)
    
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "persons"


class FaceEmbedding(BaseModel):
    """One face embedding per registered photo of a person."""

    person = ForeignKeyField(
        Person,
        backref="embeddings",
        on_delete="CASCADE",
    )

    # face_recognition library (dlib) -> 128-dim embeddings
    embedding = VectorField(
        dimensions=128
    )

    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "face_encodings"

class EntryLog(BaseModel):
    """Log of a person's entry."""

    person = ForeignKeyField(
        Person,
        backref="entry_logs",
        on_delete="CASCADE",
    )

    entry_date = DateField()
    entry_time = TimeField()

    class Meta:
        table_name = "entry_logs"
# =========================
# Init
# =========================

def init_db():
    """Connect, enable the pgvector extension and create the tables."""
    db.connect()

    # The "vector" column type only exists after this extension.
    db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")

    # Both tables are listed: peewee does NOT create the Person
    # table automatically just because it is referenced by a FK.
    db.create_tables([Person, FaceEmbedding, EntryLog], safe=True)

    db.close()


if __name__ == "__main__":
    init_db()
    print("Tables ready: persons, face_encodings, entrylogs")
