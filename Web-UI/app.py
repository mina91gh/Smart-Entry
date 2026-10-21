
from datetime import date
from pathlib import Path
import sys

import os
import cv2

from fastapi.responses import StreamingResponse

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uuid
import base64
import re
import numpy as np
from threading import Lock

from fastapi import Form, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from peewee import IntegrityError

# Locate the existing Face-Recognition folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACE_RECOGNITION_DIR = PROJECT_ROOT / "Face-Recognition"
sys.path.insert(0, str(FACE_RECOGNITION_DIR))

from model import db, Person, EntryLog, FaceEmbedding512

app = FastAPI(title="Smart Entry")

BASE_DIR = Path(__file__).resolve().parent
SMART_ENTRY_RTSP_URL= (
    "rtsp://root:root@192.168.10.176:554/axis-media/media.amp"
)
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

print("BASE_DIR:", BASE_DIR.resolve())
print("TEMPLATE DIR:", (BASE_DIR / "templates").resolve())
print("PEOPLE TEMPLATE:", (BASE_DIR / "templates" / "people.html").resolve())
print("TEMPLATE EXISTS:", (BASE_DIR / "templates" / "people.html").exists())
def generate_camera_frames(cap):
    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    #camera_url = os.getenv("SMART_ENTRY_RTSP_URL", "").strip()
    camera_url = SMART_ENTRY_RTSP_URL.strip()

    if not camera_url:
        print("ERROR: SMART_ENTRY_RTSP_URL is empty!")
        raise HTTPException(
            status_code=503,
            detail="Camera RTSP URL is not configured.",
        )

    print("RTSP URL is configured.")
    cap = cv2.VideoCapture(
        camera_url,
        cv2.CAP_FFMPEG,
    )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not connect to the camera!")
        cap.release()

        raise HTTPException(
            status_code=503,
            detail="Could not connect to the camera.",
        )
    print("Camera connected successfully!")
    return StreamingResponse(
        generate_camera_frames(cap),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )




def get_entry_logs(query):
    logs = []

    for log in query:
        logs.append({
            "person_id": log.person.id,
            "profile_image": log.person.profile_image,
            "name": log.person.name,
            "entry_date": log.entry_date,
            "entry_time": log.entry_time,
            "exit_time": log.exit_time,
        })

    return logs


@app.get("/")
def dashboard(request: Request):
    registered_people = 0
    todays_entries = 0
    db_status = "Disconnected"
    entry_logs = []

    try:
        with db.connection_context():
            registered_people = Person.select().count()

            todays_entries = (
                EntryLog.select()
                .where(EntryLog.entry_date == date.today())
                .count()
            )

            logs_query = (
                EntryLog.select(EntryLog, Person)
                .join(Person)
                .where(EntryLog.entry_date == date.today())
                .order_by(EntryLog.entry_time.asc())
            )

            all_logs = get_entry_logs(logs_query)

            # Keep first entry and last exit for each person
            daily_logs = {}

            for log in all_logs:
                person_id = log["person_id"]

                if person_id not in daily_logs:
                    daily_logs[person_id] = log.copy()
                else:
                    current_log = daily_logs[person_id]

                    # Keep the earliest entry time
                    current_log["entry_time"] = min(
                        current_log["entry_time"],
                        log["entry_time"],
                    )

                    # Keep the latest exit time
                    if log["exit_time"] is not None:
                        if (
                            current_log["exit_time"] is None
                            or log["exit_time"] > current_log["exit_time"]
                        ):
                            current_log["exit_time"] = log["exit_time"]

            entry_logs = sorted(
                daily_logs.values(),
                key=lambda log: log["entry_time"],
                reverse=True,
            )

            db_status = "Connected"
    except Exception as error:
        print(f"Dashboard database error: {error}")

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "registered_people": registered_people,
            "todays_entries": todays_entries,
            "db_status": db_status,
            "entry_logs": entry_logs,
        },
    )

'''
@app.get("/people")
def people_page(request: Request):
    people = []

    try:
        with db.connection_context():
            people = list(
                Person.select().order_by(Person.name)
            )
    except Exception as error:
        print(f"People page database error: {error}")

    return templates.TemplateResponse(
        request=request,
        name="people.html",
        context={"people": people},
    )

'''
@app.get("/people")
def people_page(request: Request):
    people = []

    try:
        with db.connection_context():
            people = list(
                Person.select().order_by(Person.name)
            )
    except Exception as error:
        print(f"People page database error: {error}")

    print("People count:", len(people))
    print("Template path:", (BASE_DIR / "templates" / "people.html").resolve())
    template_path = BASE_DIR / "templates" / "people.html"
    print("Template contains form:", "photo-upload-form" in template_path.read_text(encoding="utf-8"))
    return templates.TemplateResponse(
        request=request,
        name="people.html",
        context={"people": people},
    )
@app.post("/people/{person_id}/profile-image")
async def update_person_profile_image(
    person_id: int,
    profile_image: UploadFile = File(...),
):
    profile_dir = BASE_DIR / "static" / "uploads" / "people"
    new_profile_path = None

    try:
        with db.connection_context():
            person = Person.get_or_none(Person.id == person_id)

        if person is None:
            raise HTTPException(
                status_code=404,
                detail="Person not found.",
            )

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
        }

        if profile_image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Choose a JPEG, PNG, or WebP image.",
            )

        # Read at most 5 MB plus one byte to enforce the limit.
        raw = await profile_image.read(5 * 1024 * 1024 + 1)

        if not raw:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="The profile image must be 5 MB or smaller.",
            )

        # Decode and validate the actual image, not just its filename.
        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            )

        # Re-encode as JPEG so uploaded files have a consistent format.
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Could not process the uploaded image.",
            )

        profile_dir.mkdir(parents=True, exist_ok=True)

        new_filename = f"{uuid.uuid4().hex}.jpg"
        new_profile_path = profile_dir / new_filename

        new_profile_path.write_bytes(encoded.tobytes())

        old_profile_url = person.profile_image

        # Update only the profile_image field; face embeddings and
        # attendance records are not modified.
        try:
            with db.connection_context():
                with db.atomic():
                    person.profile_image = (
                        f"/static/uploads/people/{new_filename}"
                    )
                    person.save(only=[Person.profile_image])

        except Exception:
            new_profile_path.unlink(missing_ok=True)
            new_profile_path = None
            raise

        # Remove the old image only if it was inside our profile folder.
        if old_profile_url:
            old_prefix = "/static/uploads/people/"

            if old_profile_url.startswith(old_prefix):
                old_filename = old_profile_url[len(old_prefix):]

                # Only allow a plain filename, never a path supplied by a user.
                if (
                    old_filename
                    and Path(old_filename).name == old_filename
                    and old_filename != new_filename
                ):
                    old_path = profile_dir / old_filename

                    try:
                        old_path.unlink(missing_ok=True)
                    except OSError as error:
                        print(
                            f"Could not delete old profile image: {error}",
                            flush=True,
                        )

        return RedirectResponse(
            url="/people",
            status_code=303,
        )

    except HTTPException:
        raise

    except Exception as error:
        if new_profile_path is not None:
            try:
                new_profile_path.unlink(missing_ok=True)
            except OSError:
                pass

        print(
            f"Profile image upload error: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Profile photo update failed. Check the server terminal.",
        )

    finally:
        await profile_image.close()


@app.get("/people/new")
def add_person_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="add_person.html",
        context={"error": None},
    )


# FaceAnalysis is loaded only when someone is registered through the web UI.
# The browser captures frames from /video_feed, which is the configured RTSP network camera.
_registration_model = None
_registration_model_lock = Lock()
MIN_REGISTRATION_SAMPLES = 5
MAX_REGISTRATION_SAMPLES = 20
MAX_FRAME_BYTES = 5 * 1024 * 1024
DATASET_DIR = PROJECT_ROOT / "Face-Collection" / "dataset"


def get_registration_model():
    global _registration_model
    with _registration_model_lock:
        if _registration_model is None:
            from insightface.app import FaceAnalysis

            # Keep registration stable and independent from the live-recognition process.
            # GPU can be enabled later after the ONNX CUDA dependencies are confirmed.
            model = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            model.prepare(ctx_id=-1, det_size=(640, 640))
            _registration_model = model
    return _registration_model


def _decode_camera_frame(data_url: str):
    if not isinstance(data_url, str) or not data_url.startswith("data:image/jpeg;base64,"):
        raise ValueError("A captured JPEG frame is invalid. Capture it again.")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except Exception as exc:
        raise ValueError("A captured frame could not be decoded.") from exc
    if not raw:
        raise ValueError("A captured frame is empty.")
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("Each captured frame must be 5 MB or smaller.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("A captured frame is not a valid image.")
    return raw, image


def _decode_profile_image(data_url: str) -> bytes:
    allowed_prefixes = (
        "data:image/jpeg;base64,",
        "data:image/png;base64,",
        "data:image/webp;base64,",
    )

    if not isinstance(data_url, str) or not data_url.startswith(
        allowed_prefixes
    ):
        raise ValueError("Choose a valid JPEG, PNG, or WebP profile image.")

    try:
        encoded_data = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded_data, validate=True)
    except Exception as exc:
        raise ValueError("The profile image could not be decoded.") from exc

    if not raw:
        raise ValueError("The profile image is empty.")

    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("The profile image must be 5 MB or smaller.")

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError("The selected profile image is invalid.")

    # Always save the result as a JPEG, regardless of the source format.
    success, encoded_image = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise ValueError("Could not convert the profile image to JPEG.")

    return encoded_image.tobytes()

@app.post("/people/new")
async def create_person_from_network_camera(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid registration request.",
        )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid registration request.",
        )

    person_name = str(payload.get("name", "")).strip()
    samples = payload.get("samples", [])

    profile_image_data = payload.get("profileImage")
    profile_image_bytes = None

    if profile_image_data:
        try:
            profile_image_bytes = _decode_profile_image(
                profile_image_data
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            )

    if not person_name:
        raise HTTPException(
            status_code=400,
            detail="Please enter a person's name.",
        )

    if len(person_name) > 100:
        raise HTTPException(
            status_code=400,
            detail="Name must be 100 characters or fewer.",
        )

    if (
        not isinstance(samples, list)
        or not MIN_REGISTRATION_SAMPLES
        <= len(samples)
        <= MAX_REGISTRATION_SAMPLES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Capture between {MIN_REGISTRATION_SAMPLES} "
                f"and {MAX_REGISTRATION_SAMPLES} samples."
            ),
        )

    # Create a safe folder name for the dataset.
    safe_name = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", person_name)
        .strip("._-")
        or "person"
    )

    person_dir = DATASET_DIR / safe_name

    saved_files = []
    person = None
    created_person = False

    try:
        # Check whether this person is already registered.
        with db.connection_context():
            if Person.get_or_none(Person.name == person_name):
                raise HTTPException(
                    status_code=409,
                    detail="A person with this name already exists.",
                )

        # Load InsightFace only when registration is requested.
        face_app = get_registration_model()
        processed = []

        for index, data_url in enumerate(samples, start=1):
            try:
                raw, image = _decode_camera_frame(data_url)
            except ValueError as error:
                print(
                    f"Skipping invalid sample {index}: {error}",
                    flush=True,
                )
                continue

            # Add padding around the camera frame.
            height, width = image.shape[:2]
            padding = int(max(height, width) * 0.25)

            padded = cv2.copyMakeBorder(
                image,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_REPLICATE,
            )

            faces = face_app.get(padded)

            # Accept only frames containing exactly one detected face.
            if len(faces) != 1:
                print(
                    f"Skipping sample {index}: "
                    f"detected {len(faces)} faces; expected 1.",
                    flush=True,
                )
                continue

            embedding = np.asarray(
                faces[0].embedding,
                dtype=np.float32,
            )

            if embedding.shape != (512,):
                print(
                    f"Skipping sample {index}: "
                    f"invalid embedding shape {embedding.shape}.",
                    flush=True,
                )
                continue

            processed.append(
                (raw, embedding.tolist(), index)
            )

        print(
            f"Valid samples: {len(processed)} / {len(samples)}",
            flush=True,
        )

        if len(processed) < MIN_REGISTRATION_SAMPLES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Only {len(processed)} valid face samples "
                    f"found out of {len(samples)} captured. "
                    f"At least {MIN_REGISTRATION_SAMPLES} valid "
                    "samples are required. Keep your face clearly "
                    "visible and capture again."
                ),
            )

        # Ensure the dataset root and this person's folder exist.
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        person_dir.mkdir(parents=True, exist_ok=True)

        # Save the profile photo separately.
        '''
        profile_dir = BASE_DIR / "static" / "uploads" / "people"
        profile_dir.mkdir(parents=True, exist_ok=True)

        profile_name = f"{uuid.uuid4().hex}.jpg"
        profile_path = profile_dir / profile_name

        # Use the first valid camera sample as the profile image.
        profile_path.write_bytes(processed[0][0])
        saved_files.append(profile_path)
        '''

        
        # Save the selected profile image, or fall back to the camera.
        profile_dir = BASE_DIR / "static" / "uploads" / "people"
        profile_dir.mkdir(parents=True, exist_ok=True)

        profile_name = f"{uuid.uuid4().hex}.jpg"
        profile_path = profile_dir / profile_name

        if profile_image_bytes is not None:
            profile_path.write_bytes(profile_image_bytes)
        else:
            profile_path.write_bytes(processed[0][0])

        saved_files.append(profile_path)


        # Create the database record and embeddings transactionally.
        with db.connection_context():
            with db.atomic():
                person = Person.create(
                    name=person_name,
                    profile_image=f"/static/uploads/people/{profile_name}",
                )
                created_person = True

                for raw, embedding, index in processed:
                    filename = f"camera_{index:02d}.jpg"
                    target = person_dir / filename

                    # Never overwrite a file from an earlier registration.
                    suffix = index
                    while target.exists():
                        suffix += 1
                        filename = f"camera_{suffix:02d}.jpg"
                        target = person_dir / filename

                    target.write_bytes(raw)
                    saved_files.append(target)

                    FaceEmbedding512.create(
                        person=person,
                        image_name=f"{person.id}_{filename}",
                        embedding=embedding,
                    )

        print(
            f"Registered {person_name}: "
            f"{len(processed)} samples saved to {person_dir}",
            flush=True,
        )

        return {
            "ok": True,
            "message": (
                f"{person_name} registered with "
                f"{len(processed)} camera samples."
            ),
            "redirect": "/people",
        }

    except HTTPException:
        # Remove files created during an unsuccessful registration.
        for path in saved_files:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(
                    f"File cleanup error: {cleanup_error}",
                    flush=True,
                )
        raise

    except IntegrityError:
        for path in saved_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        raise HTTPException(
            status_code=409,
            detail="A person with this name already exists.",
        )

    except Exception as error:
        print(
            f"Camera registration error: {error}",
            flush=True,
        )

        # Remove files written by this registration attempt.
        for path in saved_files:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(
                    f"File cleanup error: {cleanup_error}",
                    flush=True,
                )

        # If the database transaction failed, remove any remaining
        # person/embedding records from this unsuccessful attempt.
        if created_person and person is not None:
            try:
                with db.connection_context():
                    with db.atomic():
                        FaceEmbedding512.delete().where(
                            FaceEmbedding512.person == person
                        ).execute()

                        Person.delete().where(
                            Person.id == person.id
                        ).execute()

            except Exception as cleanup_error:
                print(
                    f"Database cleanup error: {cleanup_error}",
                    flush=True,
                )

        raise HTTPException(
            status_code=500,
            detail=(
                "Registration failed. "
                "Check the web server terminal."
            ),
        )

@app.get("/people/{person_id}")
def person_detail(
    request: Request,
    person_id: int,
    filter_date: date | None = None,
):
    print("HISTORY ROUTE REACHED!", flush=True)
    print(">>> PERSON DETAIL ROUTE CALLED <<<", flush=True)
    print(">>> PERSON ID:", person_id, flush=True)
    print(">>> FILTER DATE:", filter_date, flush=True)

    person = None
    entry_logs = []

    try:
        with db.connection_context():
            person = Person.get_or_none(Person.id == person_id)

            if person is not None:
                logs_query = EntryLog.select().where(
                    EntryLog.person == person
                )

                # Filter records by the selected date.
                if filter_date is not None:
                    logs_query = logs_query.where(
                        EntryLog.entry_date == filter_date
                    )

                logs_query = logs_query.order_by(
                    EntryLog.entry_date.desc(),
                    EntryLog.entry_time.desc(),
                )

                entry_logs = get_entry_logs(logs_query)

                print("Person:", person.name, flush=True)
                print("Today's date:", date.today(), flush=True)
                print("History records:", len(entry_logs), flush=True)

                for log in entry_logs:
                    print(
                        log["entry_date"],
                        log["entry_time"],
                        log["exit_time"],
                        flush=True,
                    )

    except Exception as error:
        print(f"Person detail database error: {error}", flush=True)

    if person is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail="Person not found",
        )

    return templates.TemplateResponse(
        request=request,
        name="person_detail.html",
        context={
            "person": person,
            "entry_logs": entry_logs,
            "total_entries": len(entry_logs),
            "filter_date": filter_date,
        },
    )


