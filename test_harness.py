# Test harness for Sunday. Type a message or use !listen to speak.
# Run with: python test_harness.py
# Make sure uvicorn is running first: uvicorn app.main:app --reload

import httpx

BASE_URL = "http://localhost:8000"


def chat(message: str):
    response = httpx.post(f"{BASE_URL}/chat", json={"message": message}, timeout=60)
    response.raise_for_status()
    print(f"Sunday: {response.json()['reply']}\n")


def listen():
    print("Press Enter then speak immediately...")
    input()
    print("🎙  Listening — speak now!")
    response = httpx.post(f"{BASE_URL}/listen", timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"[heard]: {data['transcript']}")
    print(f"Sunday: {data['reply']}")
    fa = data.get("first_audio_ms")
    if fa is not None:
        print(f"[latency] stop-talking → first-audio: {fa} ms")
    print()


def wake_mode():
    response = httpx.post(f"{BASE_URL}/wake/start", timeout=10)
    response.raise_for_status()
    print("Wake word mode ON. Say 'Sunday' to activate her.")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        httpx.post(f"{BASE_URL}/wake/stop", timeout=10)
        print("\nWake word mode OFF.\n")


def main():
    print("Sunday test harness")
    print("  Type a message and press Enter  → text mode")
    print("  Type !listen and press Enter    → voice mode")
    print("  Type !wake and press Enter      → wake word mode (say 'Sunday')\n")

    while True:
        try:
            message = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not message:
            continue

        try:
            if message == "!listen":
                listen()
            elif message == "!wake":
                wake_mode()
            else:
                chat(message)
        except httpx.ConnectError:
            print("Error: Could not connect. Is uvicorn running?\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
