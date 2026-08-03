from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from .models import CameraFrame, Detection, Pose, RemoteInput, VelocityCommand


class CameraPort(Protocol):
    async def latest_frame(self) -> CameraFrame: ...


class DetectorPort(Protocol):
    async def detect(self, frame: CameraFrame) -> Sequence[Detection]: ...


class PosePort(Protocol):
    async def latest_pose(self) -> Pose: ...


class MotionPort(Protocol):
    async def command(self, command: VelocityCommand) -> None: ...

    async def stop(self) -> None: ...

    async def sit(self) -> None: ...

    async def stand(self) -> None: ...


class MediaPort(Protocol):
    async def bark(self) -> None: ...


RemoteInputHandler = Callable[[RemoteInput], Awaitable[None]]


class RemoteInputPort(Protocol):
    async def watch(self, handler: RemoteInputHandler) -> None: ...
