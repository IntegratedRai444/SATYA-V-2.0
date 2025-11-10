from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/")
async def chat_with_assistant(request: Request):
    data = await request.json()
    user_message = data.get("message")
    # For now, return a mock response. Replace with OpenAI/LLM call for production.
    if not user_message:
        return JSONResponse(content={"reply": "Please enter a question."})
    reply = (
        "I'm SatyaAI's assistant! I can help you interpret deepfake results, explain reports, or guide you through the analysis process. "
        "For example, you can ask: 'How do I interpret my image report?' or 'What does GAN artifact mean?'"
    )
    return JSONResponse(content={"reply": reply}) 