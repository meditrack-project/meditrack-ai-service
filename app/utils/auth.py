from fastapi import Request, HTTPException, status

def get_current_user_id(request: Request) -> str:
    """
    Extract user_id from the X-User-ID header injected by KGateway.
    Backend services no longer validate JWTs themselves.
    """
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"success": False, "message": "Missing X-User-ID header. Unauthorized."},
        )
    return user_id
