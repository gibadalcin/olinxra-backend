from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, Literal, Dict, Any
from datetime import datetime


class ButtonAction(BaseModel):
    # type: 'link' for href navigation, 'callback' for internal action name
    type: Literal['link', 'callback'] = 'link'
    href: Optional[str] = None  # can be an external URL or internal route
    target: Optional[Literal['_self', '_blank']] = '_self'
    # optional metadata for callback actions (ex: function name and params)
    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

    @validator('href')
    def href_or_name_required(cls, v, values):
        t = values.get('type')
        if t == 'link' and not v:
            raise ValueError('href is required when action.type == "link"')
        if t == 'callback' and not values.get('name'):
            raise ValueError('name is required when action.type == "callback"')
        return v


class AnalyticsSpec(BaseModel):
    event_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class VisibilitySpec(BaseModel):
    roles: Optional[list[str]] = None  # list of roles that can see the button
    min_app_version: Optional[str] = None


class ButtonBlock(BaseModel):
    # tipo should be either 'botao_destaque' or 'botao_default'
    tipo: Literal['botao_destaque', 'botao_default']
    label: str = Field(..., min_length=1, max_length=200)
    action: ButtonAction
    variant: Optional[Literal['primary', 'secondary', 'tertiary']] = 'primary'
    color: Optional[str] = None  # hex or css color
    icon: Optional[str] = None  # icon name or svg reference
    size: Optional[Literal['small', 'medium', 'large']] = 'medium'
    disabled: Optional[bool] = False
    aria_label: Optional[str] = None
    analytics: Optional[AnalyticsSpec] = None
    visibility: Optional[VisibilitySpec] = None
    position: Optional[Literal['left', 'center', 'right']] = 'center'
    created_at: Optional[datetime] = None
    temp_id: Optional[str] = None

    @validator('color')
    def validate_color(cls, v):
        if v is None:
            return v
        # simple hex color check
        if isinstance(v, str) and (v.startswith('#') and (len(v) in (4, 7))):
            return v
        # allow css color names too
        if isinstance(v, str) and v.isalpha():
            return v
        raise ValueError('color must be hex (e.g. #fff or #ffffff) or a css color name')


def validate_button_block_payload(payload: dict) -> ButtonBlock:
    """Helper to validate a single button block payload in backend code.

    Returns a ButtonBlock instance or raises pydantic.ValidationError.
    """
    return ButtonBlock.parse_obj(payload)
