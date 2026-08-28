"""Bearer-авторизация: отказ до любого инференса.

Режим задаётся AUTH_REQUIRED: по умолчанию токен обязателен. При
AUTH_REQUIRED=0 (доверенная LAN и клиент без настройки заголовков, например
OpenWhispr Self-Hosted не конфигурирует Authorization header) запросы принимаются и без Authorization.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request


async def require_token(request: Request) -> None:
    settings = request.app.state.settings
    if not settings.auth_required:
        return
    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise HTTPException(status_code=401, detail={
            "message": "требуется заголовок Authorization: Bearer <token>",
        })
    if not secrets.compare_digest(presented.strip(), settings.token):
        raise HTTPException(status_code=401, detail={"message": "неверный токен"})
