import argparse
import asyncio
import json
import time

import av
import cv2
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, VideoStreamTrack

class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, camera_index=0, width=640, height=480, fps=20):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_ts = time.time()

        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened")

    async def recv(self):
        now = time.time()
        elapsed = now - self.last_ts
        wait = self.frame_interval - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_ts = time.time()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts, video_frame.time_base = await self.next_timestamp()
        return video_frame

    def close(self):
        if self.cap:
            self.cap.release()

async def run(args):
    ice_servers = [
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        
    ]
    config = RTCConfiguration(iceServers=ice_servers)

    pc = RTCPeerConnection(configuration=config)

    video_track = CameraVideoTrack(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    pc.addTrack(video_track)

    channel = pc.createDataChannel("telemetry")

    async def send_fake_gps():
        lat = 39.9208
        lon = 32.8541
        while True:
            if channel.readyState == "open":
                payload = {
                    "timestamp": time.time(),
                    "lat": lat,
                    "lon": lon,
                    "status": "streaming",
                }
                channel.send(json.dumps(payload))
                lat += 0.00001
                lon += 0.00001
            await asyncio.sleep(1)

    @pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        print("ICE state:", pc.iceConnectionState)

    signaling_uri = f"ws://{args.signal_host}:{args.signal_port}"

    async with websockets.connect(signaling_uri, max_size=10_000_000) as ws:
        await ws.send(json.dumps({
            "type": "register",
            "room": args.room,
            "role": "drone",
        }))

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        await ws.send(json.dumps({
            "type": "offer",
            "sdp": pc.localDescription.sdp,
            "sdpType": pc.localDescription.type,
        }))

        gps_task = asyncio.create_task(send_fake_gps())

        try:
            async for raw in ws:
                msg = json.loads(raw)

                if msg["type"] == "answer":
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=msg["sdp"], type=msg["sdpType"])
                    )

                elif msg["type"] == "candidate":
                    # aiortc candidate handling bu minimal örnekte SDP exchange sonrası çoğu durumda yeterli olur
                    pass

        finally:
            gps_task.cancel()
            await pc.close()
            video_track.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-host", required=True, help="Public signaling server IP/domain")
    parser.add_argument("--signal-port", type=int, default=8765)
    parser.add_argument("--room", default="forestfire-room")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--turn-host", default=None)
    parser.add_argument("--turn-user", default=None)
    parser.add_argument("--turn-pass", default=None)
    args = parser.parse_args()

    asyncio.run(run(args))