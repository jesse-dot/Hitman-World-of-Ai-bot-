"""
Hitman: World of Assassination — AI Agent
Uses the Gemini API (gemma-3-27b-it) to analyze live gameplay screenshots
and issue keyboard commands via pydirectinput.
"""

import io
import json
import os
import time
from typing import Any

import google.generativeai as genai
import PIL.ImageGrab
import pydirectinput
import pyttsx3
from google.api_core.exceptions import GoogleAPICallError
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemma-3-27b-it"
COUNTDOWN_SECONDS = 5
AI_PERSONALITY = os.environ.get("AI_PERSONALITY", "balanced").strip().lower()
ENABLE_TTS = os.environ.get("ENABLE_TTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

BASE_PROMPT_INSTRUCTION = (
    "Analyze this Hitman gameplay. Output ONLY a JSON object with: "
    '"thought" (string), "key" (string, e.g., "w", "a", "s", "d", "f", "c"), '
    '"duration" (float), and "hold_ctrl" (boolean for Instinct mode).'
)

PERSONALITY_INSTRUCTIONS = {
    "balanced": (
        "Play with balanced risk: move with purpose, avoid unnecessary combat, "
        "and prioritize stealth when practical."
    ),
    "cautious": (
        "Play very cautiously: prioritize stealth, avoid risky open movement, "
        "minimize exposure to NPC sightlines, and disengage from danger quickly."
    ),
    "aggressive": (
        "Play aggressively: move faster, take calculated risks, and push objectives "
        "even when stealth is partially compromised."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_prompt_instruction(personality: str) -> str:
    """Build the full prompt instruction from the selected personality."""
    personality_instruction = PERSONALITY_INSTRUCTIONS.get(personality)
    if personality_instruction is None:
        print(
            f"[Config] Unknown AI_PERSONALITY={personality!r}; "
            "falling back to 'balanced'."
        )
        personality_instruction = PERSONALITY_INSTRUCTIONS["balanced"]

    return (
        f"{BASE_PROMPT_INSTRUCTION} {personality_instruction} "
        "Consider only what is visible right now. "
        "Choose one immediate action."
    )


def create_tts_engine(enabled: bool) -> Any | None:
    """Create and configure a TTS engine when enabled."""
    if not enabled:
        return None

    try:
        return pyttsx3.init()
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS] Failed to initialize TTS engine: {exc}")
        return None


def speak_text(tts_engine: Any | None, text: str) -> Any | None:
    """Speak text with TTS if available and return a usable engine."""
    if tts_engine is None or not text:
        return tts_engine

    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_engine.stop()
        return tts_engine
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS] Failed to speak text: {exc}; reinitializing engine.")
        try:
            replacement_engine = pyttsx3.init()
            replacement_engine.say(text)
            replacement_engine.runAndWait()
            replacement_engine.stop()
            return replacement_engine
        except Exception as retry_exc:  # noqa: BLE001
            print(f"[TTS] Reinitialize failed: {retry_exc}")
            return None


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


def execute_action(action: dict, tts_engine: Any | None = None) -> Any | None:
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
    tts_engine = speak_text(
        tts_engine,
        f"{thought}. Action: {key or 'no key'} for {duration:.1f} seconds.",
    )

    if not key:
        print("No key specified — skipping action.\n")
        return tts_engine

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
    return tts_engine


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
    prompt_instruction = get_prompt_instruction(AI_PERSONALITY)
    tts_engine = create_tts_engine(ENABLE_TTS)

    print(f"[Config] AI personality: {AI_PERSONALITY}")
    print(f"[Config] TTS enabled: {tts_engine is not None}")

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
    )

    countdown(COUNTDOWN_SECONDS)

    print("Agent running — press Ctrl+C to stop.\n")

    while True:
        # 1. Capture the screen
        image = capture_screen()

        # 2. Convert to bytes for the SDK
        image_bytes = image_to_bytes(image)
        image_part = {"mime_type": "image/png", "data": image_bytes}

        try:
            # 3. Send to Gemini
            response = model.generate_content(
                [prompt_instruction, image_part]
            )
        except GoogleAPICallError as exc:
            print(f"[API error] {exc}")
            # Back off to respect rate limits before retrying
            time.sleep(5)
            continue

        response_text = response.text

        try:
            # 4. Parse JSON
            action = parse_action(response_text)
        except json.JSONDecodeError as exc:
            print(f"[JSON error] Could not parse model response: {exc}")
            print(f"  Raw response: {response_text!r}\n")
            time.sleep(1)
            continue

        # 5. Execute the action
        tts_engine = execute_action(action, tts_engine=tts_engine)


if __name__ == "__main__":
    main()
