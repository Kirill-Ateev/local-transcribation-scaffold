"""Bearer-авторизация: отказ до любого инференса."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request


async def require_token(request: Request) -> None:
    settings = request.app.state.settings
    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise HTTPException(status_code=401, detail={
            "message": "требуется заголовок Authorization: Bearer <token>",
        })
    if not secrets.compare_digest(presented.strip(), settings.token):
        raise HTTPException(status_code=401, detail={"message": "неверный токен"})
