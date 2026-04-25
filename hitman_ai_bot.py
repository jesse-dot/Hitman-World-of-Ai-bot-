"""
Hitman: World of Assassination — AI Agent
Uses the Gemini API (gemma-3-27b) to analyse live gameplay screenshots
and issue keyboard commands via pydirectinput.
"""

import io
import json
import os
import time

import google.generativeai as genai
import PIL.ImageGrab
import pydirectinput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemma-3-27b-it"
COUNTDOWN_SECONDS = 5

SYSTEM_INSTRUCTION = (
    "Analyze this Hitman gameplay. Output ONLY a JSON object with: "
    '"thought" (string), "key" (string, e.g., "w", "a", "s", "d", "f", "c"), '
    '"duration" (float), and "hold_ctrl" (boolean for Instinct mode).'
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def countdown(seconds: int) -> None:
    """Print a countdown to give the player time to Alt-Tab into the game."""
    print(f"Starting in {seconds} seconds — switch to the game now!")
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...")
        time.sleep(1)
    print("GO!\n")


def capture_screen() -> PIL.Image.Image:
    """Capture the entire primary monitor at its native resolution."""
    return PIL.ImageGrab.grab(all_screens=False)


def image_to_bytes(image: PIL.Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes for the Generative AI SDK."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def parse_action(response_text: str) -> dict:
    """
    Extract the JSON object from the model's response.

    The model is instructed to return *only* a JSON object, but it may
    occasionally wrap it in a markdown code fence.  This function strips
    any such wrapper before parsing.
    """
    text = response_text.strip()

    # Strip optional markdown code fence (```json … ``` or ``` … ```)
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line and the closing fence line
        inner_lines = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    return json.loads(text)


def execute_action(action: dict) -> None:
    """
    Execute the keyboard action described by *action*.

    Expected keys
    -------------
    key       : str   – key to press (e.g. "w", "a", "s", "d", "f", "c")
    duration  : float – how long to hold the key in seconds
    hold_ctrl : bool  – whether to hold Ctrl simultaneously (Instinct mode)
    thought   : str   – logged but not acted upon
    """
    key: str = action.get("key", "")
    duration: float = float(action.get("duration", 0.1))
    hold_ctrl: bool = bool(action.get("hold_ctrl", False))
    thought: str = action.get("thought", "")

    print(f"Thought : {thought}")
    print(f"Action  : key={key!r}, duration={duration:.2f}s, hold_ctrl={hold_ctrl}")

    if not key:
        print("No key specified — skipping action.\n")
        return

    if hold_ctrl:
        pydirectinput.keyDown("ctrl")

    try:
        pydirectinput.keyDown(key)
        time.sleep(max(duration, 0.0))
        pydirectinput.keyUp(key)
    finally:
        if hold_ctrl:
            pydirectinput.keyUp("ctrl")

    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export it before running the script."
        )

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    countdown(COUNTDOWN_SECONDS)

    print("Agent running — press Ctrl+C to stop.\n")

    while True:
        try:
            # 1. Capture the screen
            image = capture_screen()

            # 2. Convert to bytes for the SDK
            image_bytes = image_to_bytes(image)
            image_part = {"mime_type": "image/png", "data": image_bytes}

            # 3. Send to Gemini
            response = model.generate_content(
                ["Analyze the current game state and decide the next action.", image_part]
            )

            response_text = response.text

            # 4. Parse JSON
            action = parse_action(response_text)

            # 5. Execute the action
            execute_action(action)

        except json.JSONDecodeError as exc:
            print(f"[JSON error] Could not parse model response: {exc}")
            print(f"  Raw response: {response_text!r}\n")
            time.sleep(1)

        except Exception as exc:  # noqa: BLE001 – catch-all for API rate limits etc.
            error_msg = str(exc)
            print(f"[API error] {error_msg}")
            # Back off to respect rate limits before retrying
            time.sleep(5)


if __name__ == "__main__":
    main()
