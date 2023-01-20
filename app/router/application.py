from fastapi import APIRouter
from app.resources import user

router = APIRouter()

router.include_router(user)
