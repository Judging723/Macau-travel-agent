import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.agents.orchestrator import get_travel_orchestrator
from app.api.auth_routes import CurrentUser, DatabaseSession
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    chat_data: ChatRequest,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> ChatResponse:
    conversation_id = chat_data.conversation_id or uuid4()

    try:
        orchestrator = get_travel_orchestrator()

        reply = await orchestrator.answer(
            message=chat_data.message,
            user_id=current_user.id,
            conversation_id=conversation_id,
            session=session,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Travel agent request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Travel agent service is unavailable",
        ) from exc

    return ChatResponse(reply=reply, conversation_id=conversation_id)
