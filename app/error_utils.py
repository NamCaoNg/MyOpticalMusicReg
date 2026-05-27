def build_error_detail(code: str, message: str, details: object | None = None) -> dict:
    payload = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload
