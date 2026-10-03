import os

import face_recognition

from model import Person, FaceEmbedding


DATASET_PATH = "D:/Rasad/Smart-Entry- GPT/ai-service/Face-Collection/dataset"


for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue


    # 1. Get (or create) the person in the database
    person, created = Person.get_or_create(name=person_name)

    if created:
        print(f"Created person: {person_name}")


    # 2. Process each image of this person
    for image_name in os.listdir(person_folder):

        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(person_folder, image_name)

        image = face_recognition.load_image_file(image_path)


        # 3. Detect the face and compute its 128-dim encoding
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"  Skipping {image_name} - no face")
            continue


        # 4. Save the encoding of the first (largest) face
        FaceEmbedding.create(
            person=person,
            embedding=encodings[0],
        )

        print(f"  Saved: {image_name}")
