"""JWT authentication middleware for cardio-echo-suite services.

Implements OAuth2 Bearer token validation with scopes. Designed to integrate
with hospital IAM systems (Keycloak, Azure AD, AWS Cognito, Auth0).

Usage:
    from cardio_echo_core.auth import setup_auth, require_scopes, Scopes

    app = FastAPI()
    setup_auth(
        app,
        jwt_secret=os.environ["JWT_SECRET"],
        jwt_issuer=os.environ.get("JWT_ISSUER", "cardio-echo-suite"),
        jwt_audience=os.environ.get("JWT_AUDIENCE", "cardio-echo-suite"),
    )

    @app.post("/predict")
    @require_scopes(Scopes.ECHO_READ)
    async def predict(...):
        ...

Scopes (SMART-on-FHIR inspired):
- patient/*.read     — read patient data
- patient/*.write    — write patient data
- system/*.read      — system-level read (admin)
- system/*.write     — system-level write (admin)
- echo.predict       — invoke AI prediction endpoints
- ecg.predict        — invoke ECG prediction endpoints
- segment.prompt     — invoke promptable segmentation
- report.draft       — generate draft reports
- report.sign        — sign/finalize reports (clinician)
- finetune.submit    — submit fine-tune jobs (researcher)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import jwt
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class Scopes:
    """SMART-on-FHIR inspired scope strings."""
    PATIENT_READ = "patient/*.read"
    PATIENT_WRITE = "patient/*.write"
    SYSTEM_READ = "system/*.read"
    SYSTEM_WRITE = "system/*.write"
    ECHO_PREDICT = "echo.predict"
    ECG_PREDICT = "ecg.predict"
    SEGMENT_PROMPT = "segment.prompt"
    REPORT_DRAFT = "report.draft"
    REPORT_SIGN = "report.sign"
    FINETUNE_SUBMIT = "finetune.submit"


@dataclass
class AuthConfig:
    jwt_secret: str
    jwt_issuer: str = "cardio-echo-suite"
    jwt_audience: str = "cardio-echo-suite"
    jwt_algorithm: str = "HS256"
    # If True, requests with no Authorization header are allowed (dev mode).
    # NEVER set to True in production.
    allow_no_auth: bool = False
    # If set, only these scopes are accepted (whitelist).
    allowed_scopes: Optional[set] = None


# Module-level config — set by setup_auth()
_config: Optional[AuthConfig] = None
_bearer = HTTPBearer(auto_error=False)


def setup_auth(
    app: FastAPI,
    jwt_secret: str,
    jwt_issuer: str = "cardio-echo-suite",
    jwt_audience: str = "cardio-echo-suite",
    jwt_algorithm: str = "HS256",
    allow_no_auth: bool = False,
    allowed_scopes: Optional[set] = None,
) -> None:
    """Install JWT auth middleware on the FastAPI app."""
    global _config
    _config = AuthConfig(
        jwt_secret=jwt_secret,
        jwt_issuer=jwt_issuer,
        jwt_audience=jwt_audience,
        jwt_algorithm=jwt_algorithm,
        allow_no_auth=allow_no_auth,
        allowed_scopes=allowed_scopes,
    )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # Skip auth for health/docs
        if request.url.path in {"/healthz", "/readyz", "/", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        if _config is None:
            return await call_next(request)

        # Extract bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            if _config.allow_no_auth:
                request.state.user = {"sub": "anonymous", "scope": ""}
                return await call_next(request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]
        try:
            payload = jwt.decode(
                token,
                _config.jwt_secret,
                algorithms=[_config.jwt_algorithm],
                issuer=_config.jwt_issuer,
                audience=_config.jwt_audience,
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(401, "Token expired")
        except jwt.InvalidAudienceError:
            raise HTTPException(401, "Invalid audience")
        except jwt.InvalidIssuerError:
            raise HTTPException(401, "Invalid issuer")
        except jwt.InvalidTokenError as e:
            raise HTTPException(401, f"Invalid token: {e}")

        # Validate scopes if whitelist is set
        token_scopes = set(payload.get("scope", "").split())
        if _config.allowed_scopes and not token_scopes & _config.allowed_scopes:
            raise HTTPException(403, "Token has no allowed scopes")

        # Attach user info to request state
        request.state.user = {
            "sub": payload.get("sub", "unknown"),
            "scope": payload.get("scope", ""),
            "scopes": list(token_scopes),
            "name": payload.get("name"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
        }

        return await call_next(request)


def require_scopes(*required_scopes: str) -> Callable:
    """Decorator: require the caller to have any of the given scopes."""
    def decorator(fn: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # FastAPI passes the Request as the first arg of dependencies,
            # but for path operations it's accessible via kwargs["request"]
            request: Optional[Request] = kwargs.get("request")
            if request is None:
                for a in args:
                    if isinstance(a, Request):
                        request = a
                        break
            if request is None or not hasattr(request, "state"):
                # No request available — fall through (auth middleware will catch)
                return await fn(*args, **kwargs) if _is_coro(fn) else fn(*args, **kwargs)

            user = getattr(request.state, "user", None)
            if user is None:
                # allow_no_auth mode — anonymous user
                if _config and _config.allow_no_auth:
                    return await fn(*args, **kwargs) if _is_coro(fn) else fn(*args, **kwargs)
                raise HTTPException(401, "Not authenticated")

            user_scopes = set(user.get("scopes", []))
            if not user_scopes & set(required_scopes):
                raise HTTPException(
                    403,
                    f"Required scopes: {required_scopes}. Have: {user_scopes}",
                )

            return await fn(*args, **kwargs) if _is_coro(fn) else fn(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


def _is_coro(fn: Callable) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


def issue_token(
    sub: str,
    scopes: List[str],
    secret: Optional[str] = None,
    expires_in_s: int = 3600,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Issue a JWT token — used for testing and the local dev issuer."""
    secret = secret or (_config.jwt_secret if _config else "dev-secret")
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": sub,
        "scope": " ".join(scopes),
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + expires_in_s,
        "iss": _config.jwt_issuer if _config else "cardio-echo-suite",
        "aud": _config.jwt_audience if _config else "cardio-echo-suite",
    }
    if extra_claims:
        payload.update(extra_claims)
    algo = _config.jwt_algorithm if _config else "HS256"
    return jwt.encode(payload, secret, algorithm=algo)
