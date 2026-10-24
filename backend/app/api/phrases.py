"""
Namma Kannada API - Phrases Router
CRUD and community features for dialect phrases.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from app.core.config import settings
from app.core.security import get_current_user, get_optional_user
from app.models.schemas import (
    DialectRegion,
    PhraseCreate,
    PhraseResponse,
    PhraseListResponse,
)
from app.services.firestore_service import FirestoreService
from app.services.moderation_service import ModerationService
from app.services.storage_service import StorageService

router = APIRouter(prefix="/phrases", tags=["Phrases"])


@router.post("/", response_model=PhraseResponse, status_code=status.HTTP_201_CREATED)
async def create_phrase(
    phrase: PhraseCreate,
    current_user: dict = Depends(get_current_user),
):
    """Create a new dialect phrase (authenticated)."""
    firestore_svc = FirestoreService()
    moderation_svc = ModerationService()

    phrase_data = phrase.model_dump()
    phrase_data["contributor_id"] = current_user["uid"]
    phrase_data["contributor_name"] = current_user.get("display_name", "Anonymous")

    created = await firestore_svc.create_phrase(phrase_data, user_id=current_user["uid"])

    # Kick off async moderation
    await moderation_svc.moderate_submission(created["id"], phrase_data)

    return PhraseResponse(**created)


@router.get("/", response_model=PhraseListResponse)
async def list_phrases(
    region: Optional[DialectRegion] = None,
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """List phrases with optional filtering, pagination, and search (public)."""
    firestore_svc = FirestoreService()
    result = await firestore_svc.list_phrases(
        region=region,
        page=page,
        page_size=page_size,
        search=search,
    )
    return PhraseListResponse(**result)


@router.get("/{phrase_id}", response_model=PhraseResponse)
async def get_phrase(phrase_id: str):
    """Get a single phrase by ID."""
    firestore_svc = FirestoreService()
    phrase = await firestore_svc.get_phrase(phrase_id)
    if not phrase:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phrase '{phrase_id}' not found",
        )
    return PhraseResponse(**phrase)


@router.post("/{phrase_id}/upvote")
async def upvote_phrase(
    phrase_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Upvote a phrase (authenticated)."""
    firestore_svc = FirestoreService()
    updated_count = await firestore_svc.upvote_phrase(phrase_id, current_user["uid"])
    return {"phrase_id": phrase_id, "upvotes": updated_count}


@router.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload an audio file for a phrase contribution."""
    # Validate content type
    if file.content_type not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio type '{file.content_type}'. Allowed: {settings.ALLOWED_AUDIO_TYPES}",
        )

    # Validate file size
    contents = await file.read()
    max_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_AUDIO_SIZE_MB} MB",
        )

    storage_svc = StorageService()
    res = storage_svc.upload_audio(
        file_bytes=contents,
        filename=file.filename or "audio.webm",
        content_type=file.content_type or "audio/webm",
    )

    return {"audio_url": res.get("public_url") or res.get("file_path")}


@router.get("/dialects/{region_id}", response_model=PhraseListResponse)
async def get_dialect_phrases(region_id: str):
    """Retrieve verified linguistic entries associated with the requested region."""
    firestore_svc = FirestoreService()
    result = await firestore_svc.list_phrases(region=region_id, page=1, page_size=50)
    return PhraseListResponse(**result)


@router.post("/contributions/new", response_model=PhraseResponse, status_code=status.HTTP_201_CREATED)
async def create_contribution(
    phrase: PhraseCreate,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Data Ingestion: New crowdsourced contribution portal endpoint."""
    uid = current_user["uid"] if current_user else "field-contributor"
    name = current_user.get("display_name", "Field Volunteer") if current_user else "Field Volunteer"
    firestore_svc = FirestoreService()
    moderation_svc = ModerationService()
    phrase_data = phrase.model_dump()
    phrase_data["contributor_id"] = uid
    phrase_data["contributor_name"] = name
    phrase_data["moderation_status"] = "pending"
    created = await firestore_svc.create_phrase(phrase_data, user_id=uid)
    await moderation_svc.moderate_submission(created["id"], phrase_data)
    return PhraseResponse(**created)
