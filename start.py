"""One-command launcher for the NileMart project.

    python start.py                 # menu
    python start.py app             # the app (overview, notebook walkthrough, model comparison, planner)
    python start.py walkthrough     # the app, opened on the notebook walkthrough
    python start.py compare         # the app, opened on the model comparison
    python start.py planner         # the app, opened on the weekly planner
    python start.py api             # REST API (docs at /docs)
    python start.py all             # app + API together
    python start.py train           # retrain the model
    python start.py test            # run the tests

On the first run it installs the requirements and trains the model if needed.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
APP_URL = "http://localhost:8501"

SERVICES = {
    "app": ("NileMart app", [PY, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501",
                             "--server.headless", "true"]),
    "api": ("Forecast API", [PY, "-m", "uvicorn", "api.main:app", "--port", "8000"]),
}

# option -> (services to start, pages to open)
TARGETS = {
    "app": (["app"], [APP_URL]),
    "walkthrough": (["app"], [f"{APP_URL}/notebook"]),
    "compare": (["app"], [f"{APP_URL}/compare"]),
    "planner": (["app"], [f"{APP_URL}/planner"]),
    "api": (["api"], ["http://localhost:8000/docs"]),
    "all": (["app", "api"], [APP_URL, "http://localhost:8000/docs"]),
}


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, env=ENV, **kw)


def ensure_requirements():
    try:
        import fastapi, sklearn, streamlit, uvicorn  # noqa: F401
    except ImportError:
        print("Installing requirements (first run only)...")
        r = run([PY, "-m", "pip", "install", "-r", "requirements.txt"])
        if r.returncode != 0:
            sys.exit("Could not install the requirements. Run: pip install -r requirements.txt")


def ensure_model():
    if not (ROOT / "models/model.json").exists():
        print("Training the model (first run only)...")
        run([PY, "-m", "nilemart.train"], check=True)


def launch(target):
    services, urls = TARGETS[target]
    procs = []
    for s in services:
        title, cmd = SERVICES[s]
        print(f"Starting {title}...")
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=ENV))
    time.sleep(4)
    for url in urls:
        print(f"Opening {url}")
        webbrowser.open(url)
    print("\nRunning. Press Ctrl+C to stop.")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        print("Stopped.")


def menu() -> str:
    options = [("app", "NileMart app (overview)"), ("walkthrough", "Notebook walkthrough"),
               ("compare", "Model comparison"), ("planner", "Weekly planner"), ("api", "Forecast API"),
               ("all", "App + API"), ("train", "Retrain the model"), ("test", "Run the tests")]
    print("\nNileMart Sales Forecast\n")
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i}. {label}")
    choice = input(f"\nChoose 1-{len(options)} [1]: ").strip() or "1"
    try:
        return options[int(choice) - 1][0]
    except (ValueError, IndexError):
        sys.exit(f"Please choose a number from 1 to {len(options)}.")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else menu()
    ensure_requirements()
    if task == "train":
        run([PY, "-m", "nilemart.train"], check=True)
        return
    if task == "test":
        sys.exit(run([PY, "-m", "pytest", "-q"]).returncode)
    if task not in TARGETS:
        sys.exit(f"Unknown option '{task}'. Use: {', '.join(TARGETS)}, train or test.")
    ensure_model()
    launch(task)


if __name__ == "__main__":
    main()
