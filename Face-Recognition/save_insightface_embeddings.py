
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from model import db, Person, FaceEmbedding512


# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "Face-Collection" / "dataset"

# Image preprocessing
PADDING_RATIO = 0.35

# Use CPU for now because GPU inference has a cuDNN loading error
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))


def generate_embedding(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image")

    height, width = image.shape[:2]
    padding = int(max(height, width) * PADDING_RATIO)

    padded_image = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_REPLICATE,
    )

    faces = app.get(padded_image)

    if not faces:
        raise ValueError("No face detected")

    # Select the largest detected face
    face = max(
        faces,
        key=lambda item: (
            item.bbox[2] - item.bbox[0]
        ) * (
            item.bbox[3] - item.bbox[1]
        ),
    )

    embedding = face.embedding

    if embedding.shape != (512,):
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    return embedding.astype(np.float32).tolist()


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    created = 0
    skipped = 0
    failed = 0

    db.connect(reuse_if_open=True)

    try:
        for person_dir in sorted(DATASET_DIR.iterdir()):
            if not person_dir.is_dir():
                continue

            try:
                person = Person.get(
                    Person.name == person_dir.name
                )
            except Person.DoesNotExist:
                print(
                    f"[ERROR] Person not found in database: "
                    f"{person_dir.name}"
                )
                failed += 1
                continue

            images = sorted(
                path
                for path in person_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp"}
            )

            print(
                f"\nProcessing {person.name}: "
                f"{len(images)} images"
            )

            for image_path in images:
                try:
                    # Avoid inserting the same image twice
                    existing = FaceEmbedding512.get_or_none(
                        (FaceEmbedding512.person == person)
                        & (
                            FaceEmbedding512.image_name
                            == image_path.name
                        )
                    )

                    if existing:
                        skipped += 1
                        continue

                    embedding = generate_embedding(image_path)

                    with db.atomic():
                        FaceEmbedding512.create(
                            person=person,
                            image_name=image_path.name,
                            embedding=embedding,
                        )

                    created += 1
                    print(f"[SAVED] {person.name}/{image_path.name}")

                except Exception as error:
                    failed += 1
                    print(
                        f"[FAILED] {person_dir.name}/"
                        f"{image_path.name}: {error}"
                    )

        print("\n========== FINAL REPORT ==========")
        print(f"Created: {created}")
        print(f"Skipped: {skipped}")
        print(f"Failed:  {failed}")
        print("==================================")

    finally:
        if not db.is_closed():
            db.close()


if __name__ == "__main__":
    main()
