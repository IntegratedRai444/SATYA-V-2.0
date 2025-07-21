from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/login")
async def login(request: Request):
    """
    Authenticates a user. Returns standardized response with success flag, token, and user info on success,
    or success: false and error message on failure.
    """
    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        print(f"Login attempt: username={username}, password={'***' if password else None}")
        if not username:
            return JSONResponse(content={"success": False, "message": "Username is required"}, status_code=400)
        if not password:
            return JSONResponse(content={"success": False, "message": "Password is required"}, status_code=400)
        # Demo: accept any non-empty username/password
        return JSONResponse(content={
            "success": True,
            "token": "demo-token",
            "user": {"username": username}
        })
    except Exception as e:
        print(f"Login error: {e}")
        return JSONResponse(content={"success": False, "message": f"Authentication system error: {str(e)}"}, status_code=500)

@router.get("/validate")
async def validate():
    return JSONResponse(content={"valid": True}) 