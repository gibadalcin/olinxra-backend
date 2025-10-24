import pytest
from bson import ObjectId
from fastapi import HTTPException

# import helpers from main
from main import format_address_string, parse_objectid


def test_format_address_string_full():
    endereco = {
        "road": "Rua X",
        "suburb": "Bairro Y",
        "city": "Cidade Z",
        "state": "Estado W",
        "country": "Brasil"
    }
    s = format_address_string(endereco)
    assert s == "Rua X, Bairro Y, Cidade Z, Estado W, Brasil"


def test_format_address_string_partial():
    endereco = {
        "road": "",
        "suburb": None,
        "town": "Pequena Vila",
        "state": "Estado",
        # country missing
    }
    s = format_address_string(endereco)
    assert s == "Pequena Vila, Estado"


def test_format_address_string_empty():
    assert format_address_string({}) == ""
    assert format_address_string(None) == ""


def test_parse_objectid_from_objectid():
    oid = ObjectId()
    out = parse_objectid(oid, 'test')
    assert isinstance(out, ObjectId)
    assert str(out) == str(oid)


def test_parse_objectid_from_str():
    oid = ObjectId()
    out = parse_objectid(str(oid), 'test')
    assert isinstance(out, ObjectId)
    assert str(out) == str(oid)


def test_parse_objectid_invalid():
    with pytest.raises(HTTPException) as exc:
        parse_objectid('not-a-valid-id', 'test')
    assert exc.value.status_code == 400
        # Removed stray patch marker