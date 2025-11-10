import logging
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import secrets

router = APIRouter()

logger = logging.getLogger("satyaai")
logging.basicConfig(level=logging.INFO)

# In-memory user store for demo (replace with DB in production)
users = {}
api_keys = {}

class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

@router.post("/register", summary="Register a new user and get an API key")
async def register(user: UserRegister):
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    users[user.username] = user.password
    api_key = secrets.token_hex(16)
    api_keys[user.username] = api_key
    logger.info(f"User registered: {user.username}")
    return {"success": True, "api_key": api_key}

@router.post("/login", summary="Login and get your API key")
async def login(user: UserLogin):
    if users.get(user.username) != user.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    api_key = api_keys.get(user.username)
    if not api_key:
        api_key = secrets.token_hex(16)
        api_keys[user.username] = api_key
    logger.info(f"User logged in: {user.username}")
    return {"success": True, "api_key": api_key}

@router.post("/key", summary="Generate a new API key (requires username/password)")
async def generate_key(user: UserLogin):
    if users.get(user.username) != user.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    api_key = secrets.token_hex(16)
    api_keys[user.username] = api_key
    logger.info(f"API key regenerated for: {user.username}")
    return {"success": True, "api_key": api_key}

@router.post("/api/auth/logout")
def logout(request: Request):
    return {"success": True, "message": "Logged out (stub)"}

@router.post("/api/auth/validate")
def validate(request: Request):
    return {"valid": True, "user": {"username": "demo"}} 