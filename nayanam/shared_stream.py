import numpy as np
from multiprocessing import shared_memory

FRAME_SHAPE = (640, 640, 3)
FRAME_DTYPE = np.uint8

MAX_DETECTIONS = 128


class SharedFrameChannel:
    """One shared-memory slot holding a single video frame."""

    def __init__(self, stream_id, shape=FRAME_SHAPE, dtype=FRAME_DTYPE,
                 create=True, shm_name=None):
        self.stream_id = stream_id
        self.shape = shape
        self.dtype = dtype
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

        if create:
            shm_name = f"frame_shm_{stream_id}"

            self.shm = shared_memory.SharedMemory(
                name=shm_name,
                create=True,
                size=nbytes
            )

            self.name = shm_name
        else:
            if shm_name is None:
                raise ValueError("shm_name is required when create=False")
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.name = shm_name

        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

    def write(self, frame):
        """Copy a frame (already resized to `shape`) into shared memory."""
        np.copyto(self.array, frame)

    def read(self):
        """Return a private copy of the current frame.

        A copy (not a view) is returned so the caller can keep using it
        after the producer overwrites the buffer on the next cycle.
        """
        return self.array.copy()

    def close(self):
        self.shm.close()

    def unlink(self):
        """Release the underlying OS shared-memory block. Call exactly
        once, from whichever process created it, after all other
        processes have called close()."""
        self.shm.unlink()


class SharedResultChannel:
    """One shared-memory slot holding raw (pre-tracking) detections:
    xyxy boxes, confidence scores, and class ids -- i.e. exactly what
    sv.Detections.from_ultralytics(results) produces, before ByteTrack
    assigns tracker ids (tracking stays in the per-camera process since
    each camera needs its own independent tracker state)."""

    def __init__(self, stream_id, max_det=MAX_DETECTIONS,
                 create=True, shm_name=None):
        self.stream_id = stream_id
        self.max_det = max_det

        # flat float32 layout: [count][xyxy: max_det*4][confidence: max_det][class_id: max_det]
        self.xyxy_size = max_det * 4
        self.conf_size = max_det
        self.cls_size = max_det
        total = 1 + self.xyxy_size + self.conf_size + self.cls_size
        nbytes = total * np.dtype(np.float32).itemsize

        if create:
            shm_name = f"result_shm_{stream_id}"

            self.shm = shared_memory.SharedMemory(
                name=shm_name,
                create=True,
                size=nbytes
            )

            self.name = shm_name
        else:
            if shm_name is None:
                raise ValueError("shm_name is required when create=False")
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.name = shm_name

        buf = np.ndarray((total,), dtype=np.float32, buffer=self.shm.buf)
        self._count = buf[0:1]
        self.xyxy = buf[1:1 + self.xyxy_size].reshape(max_det, 4)
        self.confidence = buf[1 + self.xyxy_size: 1 + self.xyxy_size + self.conf_size]
        self.class_id = buf[1 + self.xyxy_size + self.conf_size:]

    def write(self, xyxy, confidence, class_id):
        n = min(len(xyxy), self.max_det)
        if n < len(xyxy):
            # Shouldn't normally happen; guard so it never silently
            # corrupts memory if a frame has more detections than expected.
            pass
        self._count[0] = n
        if n:
            self.xyxy[:n] = xyxy[:n]
            self.confidence[:n] = confidence[:n]
            self.class_id[:n] = class_id[:n]

    def read(self):
        n = int(self._count[0])
        return (
            self.xyxy[:n].copy(),
            self.confidence[:n].copy(),
            self.class_id[:n].copy(),
        )

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
