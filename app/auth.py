"""모델 API는 MES에 로그인한 관리자만 호출한다."""

from typing import Annotated

import httpx
from fastapi import Depends, Header, HTTPException

from .config import get_settings


def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "로그인이 필요합니다.")
    try:
        res = httpx.get(
            f"{get_settings().mes_url}/api/auth/me",
            headers={"authorization": authorization},
            timeout=5,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(503, "사원 권한을 확인하지 못했습니다.") from exc
    if res.status_code == 401:
        raise HTTPException(401, "로그인이 필요합니다.")
    if res.status_code != 200 or res.json().get("role") != "ADMIN":
        raise HTTPException(403, "관리자만 접근할 수 있습니다.")


Admin = Annotated[None, Depends(require_admin)]
