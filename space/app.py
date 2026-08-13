"""Private Hugging Face Space entrypoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from jump_ui.app import create_app  # noqa: E402

demo = create_app()

if __name__ == "__main__":
    demo.launch(css=demo._jump_css, footer_links=[])
