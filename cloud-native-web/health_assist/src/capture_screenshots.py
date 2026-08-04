"""Drive the running Streamlit app and capture screenshots for the report.

Start the app first:
    .venv/bin/streamlit run src/app.py --server.port 8511
Then:
    .venv/bin/python src/capture_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from playwright.sync_api import sync_playwright

import config

URL = "http://localhost:8511"
SHOTS = config.DOCS_DIR / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


def settle(page, ms=2500):
    page.wait_for_timeout(ms)


def wait_for_idle(page, timeout=420_000):
    """Streamlit shows a RUNNING status widget while a script executes."""
    page.wait_for_timeout(1500)
    try:
        page.wait_for_selector(
            "[data-testid='stStatusWidget']", state="detached", timeout=timeout
        )
    except Exception:
        pass
    page.wait_for_timeout(1500)


def shot(page, name: str):
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print("captured", path.name)


def _scroll_to(page, y: int):
    """Scroll the Streamlit main pane.

    Streamlit renders its content inside a scrollable section rather than the
    document body, so window.scrollTo has no effect here.
    """
    page.evaluate(
        """(y) => {
            const targets = [
                document.querySelector('section.main'),
                document.querySelector('[data-testid="stMain"]'),
                document.querySelector('[data-testid="stAppViewContainer"]'),
            ].filter(Boolean);
            for (const t of targets) {
                if (t.scrollHeight > t.clientHeight) { t.scrollTop = y; return; }
            }
            window.scrollTo(0, y);
        }""",
        y,
    )
    page.wait_for_timeout(900)


def shot_from(page, name: str, y: int, height: int = 1000):
    """Viewport screenshot starting at a vertical offset.

    The full-page captures are tall, so the interesting part of a result ends up
    small on the printed page. These offset captures frame one section at a time.
    """
    _scroll_to(page, y)
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path))
    print("captured", path.name)


def click_tab(page, label: str):
    page.get_by_role("tab", name=label).click()
    page.wait_for_timeout(1200)


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(URL, wait_until="networkidle", timeout=120_000)
        wait_for_idle(page)

        # 1. Landing view with the model panel
        shot(page, "01_application_home")

        # 2. Run the dengue sample case end to end
        page.get_by_role("button", name="Run triage").click()
        wait_for_idle(page)
        shot(page, "02_episode_dengue_result")
        shot_from(page, "02a_dengue_triage_and_entities", 820)
        shot_from(page, "02b_dengue_summary_and_reply", 1650)

        # 3. Emergency case, to show the rule layer overriding the model
        _scroll_to(page, 0)
        page.get_by_role("combobox").first.click()
        page.wait_for_timeout(600)
        page.get_by_role("option", name="Cardiac emergency").click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Run triage").click()
        wait_for_idle(page)
        shot(page, "03_episode_emergency_result")
        shot_from(page, "03a_emergency_rule_override", 820)
        shot_from(page, "03b_emergency_patient_reply", 1700)
        _scroll_to(page, 0)
        page.wait_for_timeout(600)

        # 4. Knowledge base question answering
        click_tab(page, "Knowledge base QA")
        page.get_by_role("combobox").first.click()
        page.wait_for_timeout(600)
        page.get_by_role(
            "option", name="What are the fasting requirements before a lipid profile?"
        ).click()
        page.wait_for_timeout(1200)
        page.get_by_role("button", name="Answer").click()
        wait_for_idle(page)
        shot(page, "04_knowledge_base_qa")

        # 5. LLMOps metrics dashboard
        click_tab(page, "LLMOps metrics")
        settle(page, 3500)
        shot(page, "05_llmops_metrics")
        shot_from(page, "05a_metric_table", 300)
        shot_from(page, "05b_latency_charts", 1150)
        shot_from(page, "05c_threshold_calibration", 2300)
        _scroll_to(page, 0)
        page.wait_for_timeout(600)

        # 6. Fine-tuned model report
        click_tab(page, "Fine-tuned model")
        settle(page, 3000)
        shot(page, "06_finetuned_model")
        shot_from(page, "06a_finetune_metrics", 300)
        shot_from(page, "06b_confusion_matrix", 950)
        _scroll_to(page, 0)
        page.wait_for_timeout(600)

        # 7. Live classification with the fine-tuned model
        page.get_by_role("button", name="Classify").click()
        wait_for_idle(page)
        shot(page, "07_finetuned_live_prediction")

        # 8. Hugging Face API access
        click_tab(page, "API access")
        page.get_by_role(
            "button", name="Retrieve model metadata from the Hub"
        ).click()
        wait_for_idle(page)
        shot(page, "08_hf_api_access")
        shot_from(page, "08a_hf_model_metadata", 250)

        # 9. Same model served locally and through the hosted endpoint
        page.get_by_role("button", name="Run locally and remotely").click()
        wait_for_idle(page)
        shot_from(page, "09_local_vs_remote_inference", 900)

        browser.close()
    print(f"\nScreenshots written to {SHOTS}")


if __name__ == "__main__":
    main()
