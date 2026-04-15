import asyncio
import json
from collections import defaultdict

import websockets

# Yapı:
# rooms[room_id]["drone"] = websocket
# rooms[room_id]["center"] = websocket
rooms = defaultdict(dict)
lock = asyncio.Lock()


async def handler(websocket):
    room = None
    role = None

    try:
        # İlk gelen mesaj register olmalı
        raw = await websocket.recv()
        msg = json.loads(raw)

        if msg.get("type") != "register":
            await websocket.send(json.dumps({
                "type": "error",
                "message": "First message must be register"
            }))
            return

        room = msg.get("room")
        role = msg.get("role")

        if not room or role not in ["drone", "center"]:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid room or role"
            }))
            return

        async with lock:
            rooms[room][role] = websocket

        print(f"[INFO] {role} joined room '{room}'")

        await websocket.send(json.dumps({
            "type": "registered",
            "room": room,
            "role": role
        }))

        # Sonraki mesajları karşı tarafa aktar
        async for raw in websocket:
            msg = json.loads(raw)

            target_role = "center" if role == "drone" else "drone"

            async with lock:
                target_ws = rooms.get(room, {}).get(target_role)

            if target_ws:
                await target_ws.send(json.dumps(msg))
            else:
                await websocket.send(json.dumps({
                    "type": "warning",
                    "message": f"Target '{target_role}' not connected yet"
                }))

    except websockets.ConnectionClosed:
        print(f"[INFO] Connection closed: role={role}, room={room}")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if room and role:
            async with lock:
                if room in rooms and role in rooms[room]:
                    del rooms[room][role]

                if room in rooms and not rooms[room]:
                    del rooms[room]

        print(f"[INFO] Cleaned up: role={role}, room={room}")


async def main():
    print("[INFO] Signaling server started on ws://0.0.0.0:8765")
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()  # sonsuza kadar çalışsın


if __name__ == "__main__":
    asyncio.run(main())