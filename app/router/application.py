from fastapi import APIRouter
from app.controllers import user

router = APIRouter()

router.include_router(user)
