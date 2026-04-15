import argparse
import asyncio
import json
import cv2
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

async def run(args):
    ice_servers = [
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    ]
    config = RTCConfiguration(iceServers=ice_servers)
    pc = RTCPeerConnection(configuration=config)

    @pc.on("track")
    def on_track(track):
        print(f"Track received: {track.kind}")
        if track.kind == "video":
            async def show_frame():
                while True:
                    try:
                        frame = await track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        cv2.imshow("Drone Video Stream [Receiver]", img)
                        # Press 'q' to quit (works if focus is on OpenCV window)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Video track ended or error: {e}")
                        break
                cv2.destroyAllWindows()
            
            asyncio.create_task(show_frame())

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"Data channel received: {channel.label}")
        @channel.on("message")
        def on_message(message):
            print(f"[{channel.label}] {message}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        print("ICE state:", pc.iceConnectionState)

    signaling_uri = f"ws://{args.signal_host}:{args.signal_port}"
    print(f"Connecting to signaling server at {signaling_uri}")

    async with websockets.connect(signaling_uri, max_size=10_000_000) as ws:
        await ws.send(json.dumps({
            "type": "register",
            "room": args.room,
            "role": "center",
        }))

        try:
            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "registered":
                    print(f"Registered as center in room '{args.room}'. Waiting for drone...")

                elif msg_type == "offer":
                    print("Received offer from drone. Creating answer...")
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=msg["sdp"], type=msg["sdpType"])
                    )

                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)

                    await ws.send(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp,
                        "sdpType": pc.localDescription.type,
                    }))
                    print("Answer sent.")

                elif msg_type == "candidate":
                    # aiortc handles ice gathering internally if set in SDP, 
                    # simple examples usually just pass here unless using trickle ICE correctly.
                    pass

                elif msg_type == "warning":
                    print(f"Signaling Warning: {msg.get('message')}")
                elif msg_type == "error":
                    print(f"Signaling Error: {msg.get('message')}")
                    
        except websockets.ConnectionClosed:
            print("Signaling server connection closed.")
        finally:
            await pc.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Video Receiver (Center)")
    parser.add_argument("--signal-host", required=True, help="Public signaling server IP/domain")
    parser.add_argument("--signal-port", type=int, default=8765)
    parser.add_argument("--room", default="forestfire-room")
    args = parser.parse_args()

    # asyncio loop
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Receiver stopped.")
