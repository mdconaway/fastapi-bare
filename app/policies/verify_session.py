from fastapi import HTTPException, Request  # Header


async def verify_session(request: Request):  # verify_session(x_token: str = Header()):
    if not isinstance(request.session, dict):
        raise HTTPException(status_code=400, detail="session does not exist!")
