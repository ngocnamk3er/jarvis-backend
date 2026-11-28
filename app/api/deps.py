from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import TokenError, verify_token

bearer_scheme = HTTPBearer(auto_error=True)


@dataclass
class CurrentUser:
    sub: str
    email: str | None = None
    # Keycloak group paths from the token, e.g. "/departments/Kế toán/editors" (see app/core/spaces.py).
    groups: tuple[str, ...] = ()


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    try:
        payload = verify_token(creds.credentials)
    except TokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None
    groups = payload.get("groups") or []
    return CurrentUser(
        sub=payload["sub"],
        email=payload.get("email"),
        groups=tuple(g for g in groups if isinstance(g, str)) if isinstance(groups, list) else (),
    )
