from fastapi import APIRouter, Request

from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    chat_service = request.app.state.chat_service
    return chat_service.answer(payload.query)
