# Test harness for Bujji. Type a message or use !listen to speak.
# Run with: python test_harness.py
# Make sure uvicorn is running first: uvicorn app.main:app --reload

import httpx

BASE_URL = "http://localhost:8000"


def chat(message: str):
    response = httpx.post(f"{BASE_URL}/chat", json={"message": message}, timeout=60)
    response.raise_for_status()
    print(f"Bujji: {response.json()['reply']}\n")


def listen():
    print("Press Enter when ready to speak...")
    input()
    response = httpx.post(f"{BASE_URL}/listen", timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"[heard]: {data['transcript']}")
    print(f"Bujji: {data['reply']}\n")


def main():
    print("Bujji test harness")
    print("  Type a message and press Enter  → text mode")
    print("  Type !listen and press Enter    → voice mode (speak into mic)\n")

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
            else:
                chat(message)
        except httpx.ConnectError:
            print("Error: Could not connect. Is uvicorn running?\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
