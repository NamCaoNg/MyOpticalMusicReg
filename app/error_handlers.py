from http import HTTPStatus

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _status_to_code(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).name.lower()
    except ValueError:
        return "http_error"


def _build_error_payload(
    status_code: int,
    message: str,
    *,
    code: str | None = None,
    details: object | None = None) -> dict:
    payload = {
        "success": False,
        "error": {
            "code": code or _status_to_code(status_code),
            "message": message,
        },
    }

    if details is not None:
        payload["error"]["details"] = details

    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            code = detail.get("code") if isinstance(detail.get("code"), str) else None
            message = detail.get("message") if isinstance(detail.get("message"), str) else str(detail)
            details = detail.get("details")
        elif isinstance(detail, str):
            code = None
            message = detail
            details = None
        else:
            code = None
            message = "Request failed"
            details = detail

        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_payload(
                exc.status_code,
                message,
                code=code,
                details=details,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_build_error_payload(
                422,
                "Invalid request data",
                code="validation_error",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_build_error_payload(
                500,
                "Internal server error",
                code="internal_server_error",
            ),
        )
