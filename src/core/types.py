# src/core/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Any, Optional
import time, uuid

ActionType = Literal["python", "note", "stop"]

@dataclass
class Step:
    action: ActionType
    code: Optional[str] = None
    note: Optional[str] = None

@dataclass
class Observation:
    success: bool
    stdout: str = ""
    stderr: str = ""
    artifacts: List[str] = field(default_factory=list)

@dataclass
class Turn:
    step: Step
    observation: Observation
    ts: float = field(default_factory=time.time)

@dataclass
class RunLog:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: List[Turn] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)