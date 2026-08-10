import jwt

from app.core.config import settings

jwks_client = jwt.PyJWKClient(
    settings.OIDC_JWKS_URL,
    cache_keys=True,
    lifespan=settings.OIDC_JWKS_CACHE_TTL_SECONDS,
)


class TokenError(Exception):
    pass


def verify_token(token: str) -> dict:
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.OIDC_AUDIENCE,
            issuer=settings.OIDC_ISSUER,
        )
    except jwt.PyJWTError as e:
        raise TokenError(str(e)) from e
