from uuid import uuid4

import jwt
import pytest

from app.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.config import get_settings


def test_password_hashing_and_verification() -> None:
    password = "strong-password"

    first_hash = hash_password(password)
    second_hash = hash_password(password)

    assert first_hash != password
    assert first_hash != second_hash
    assert first_hash.startswith("$argon2")
    assert verify_password(password, first_hash) is True
    assert verify_password("wrong-password", first_hash) is False


def test_create_access_token_contains_expected_claims() -> None:
    user_id = uuid4()
    token = create_access_token(user_id)
    payload = jwt.decode(
        token,
        get_settings().jwt_secret_key,
        algorithms=["HS256"],
    )

    assert payload["sub"] == str(user_id)
    assert payload["exp"] > payload["iat"]


def test_decode_access_token_returns_user_id() -> None:
    user_id = uuid4()
    token = create_access_token(user_id)

    decoded_user_id = decode_access_token(token)

    assert decoded_user_id == user_id


def test_decode_access_token_rejects_tampered_token() -> None:
    token = create_access_token(uuid4())
    tampered_token = token + "x"

    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(tampered_token)
