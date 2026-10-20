
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

from fastapi import Form, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from peewee import IntegrityError

# Locate the existing Face-Recognition folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACE_RECOGNITION_DIR = PROJECT_ROOT / "Face-Recognition"
sys.path.insert(0, str(FACE_RECOGNITION_DIR))

from model import db, Person, EntryLog

app = FastAPI(title="Smart Entry")

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

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
    camera_url = os.getenv("SMART_ENTRY_RTSP_URL", "").strip()

    if not camera_url:
        raise HTTPException(
            status_code=503,
            detail="Camera RTSP URL is not configured.",
        )

    cap = cv2.VideoCapture(
        camera_url,
        cv2.CAP_FFMPEG,
    )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()

        raise HTTPException(
            status_code=503,
            detail="Could not connect to the camera.",
        )

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

@app.get("/people/new")
def add_person_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="add_person.html",
        context={"error": None},
    )


@app.post("/people/new")
async def create_person(
    request: Request,
    name: str = Form(...),
    profile_photo: UploadFile | None = File(default=None),
):
    person_name = name.strip()

    if not person_name:
        return templates.TemplateResponse(
            request=request,
            name="add_person.html",
            context={"error": "Please enter a person's name."},
            status_code=400,
        )

    upload_path = None
    saved_image_path = None

    try:
        if profile_photo and profile_photo.filename:
            allowed_types = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }

            extension = allowed_types.get(profile_photo.content_type)

            if not extension:
                return templates.TemplateResponse(
                    request=request,
                    name="add_person.html",
                    context={
                        "error": "Please upload a JPG, PNG, or WEBP image."
                    },
                    status_code=400,
                )

            upload_dir = BASE_DIR / "static" / "uploads" / "people"
            upload_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{uuid.uuid4().hex}{extension}"
            upload_path = upload_dir / filename

            # Limit the upload to 5 MB.
            image_data = await profile_photo.read(5 * 1024 * 1024 + 1)

            if len(image_data) > 5 * 1024 * 1024:
                return templates.TemplateResponse(
                    request=request,
                    name="add_person.html",
                    context={"error": "Image size must not exceed 5 MB."},
                    status_code=400,
                )

            if not image_data:
                return templates.TemplateResponse(
                    request=request,
                    name="add_person.html",
                    context={"error": "The selected image is empty."},
                    status_code=400,
                )

            with open(upload_path, "wb") as image_file:
                image_file.write(image_data)

            saved_image_path = f"/static/uploads/people/{filename}"

        with db.connection_context():
            Person.create(
                name=person_name,
                profile_image=saved_image_path,
            )

        return RedirectResponse(
            url="/people",
            status_code=303,
        )

    except IntegrityError:
        if upload_path and upload_path.exists():
            upload_path.unlink()

        return templates.TemplateResponse(
            request=request,
            name="add_person.html",
            context={"error": "A person with this name already exists."},
            status_code=400,
        )

    except Exception as error:
        print(f"Create person error: {error}")

        if upload_path and upload_path.exists():
            upload_path.unlink()

        return templates.TemplateResponse(
            request=request,
            name="add_person.html",
            context={"error": "Could not save the person. Check the server terminal."},
            status_code=500,
        )

    finally:
        if profile_photo:
            await profile_photo.close()



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


