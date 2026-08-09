# Standalone repository workflow

This project is intended to run from one ordinary standalone clone of
`https://github.com/wik-wav/festvox-speech-gui.git`. The checkout itself is
the canonical code working copy; it should not also be tracked as source files
by a containing notes or monorepo checkout.

## Canonical layout

- `src/festvox_tts/` contains the implementation and packaged resources.
- `tests/` contains core and headless GUI regression tests.
- `build_festival_voice.py`, `run_gui.py`, and `run_tests.py` are stable root
  entry points.
- `docs/` contains operating, architecture, and historical implementation
  notes.
- `generated_voices/`, rendered audio, diagnostics, caches, virtual
  environments, and machine-local configuration remain ignored local state.

Runtime paths are derived from the repository root. Generated voices therefore
default to `<checkout>/generated_voices` regardless of where the clone lives.

## First setup

From a fresh clone:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe run_gui.py
```

Copy `festvox.example.json` to the ignored `festvox.json` only when the
standalone renderer needs explicit local paths. The GUI writes its ignored
machine-local settings to `src/festvox_tts/festvox_gui/config.json`.

## Local-state boundary

Git must never stage:

- source UTAU or OpenUtau banks;
- generated Festival voices;
- local JSON configuration;
- rendered or exported audio;
- diagnostic images, caches, temporary files, or virtual environments.

Source-bank paths are inputs, not project content. Builders must treat them as
read only, and generated output must live outside the source bank. Before every
commit, inspect the staged file list and reject any source recording, private
absolute path, or generated output.

```powershell
git status --short
git diff --cached --name-only
git grep -n -I -E "[A-Za-z]:[/\\\\](Users|home)|/home/[^/]+"
```

The last command is a privacy review aid, not proof by itself; inspect any
matches to distinguish examples from machine-specific paths.

## Updating the checkout

Use fast-forward-only pulls so an unexpected divergence is visible:

```powershell
git status
git fetch origin
git pull --ff-only origin main
```

Ignored generated voices and local settings are not changed by these commands.
If an update changes dependencies, rerun the installation and environment
checks before launching the GUI.

## Making a change

Stage only named files, review them, run tests, and then push:

```powershell
git add path\to\changed-file.py tests\test_changed_behavior.py
git diff --cached
.\.venv\Scripts\python.exe run_tests.py
git commit -m "Describe the change"
git push origin main
```

Do not use broad staging as a substitute for reviewing local output. A feature
branch is recommended for changes that need review or extended verification.

## Verification

Run these checks from the repository root after moving or restoring a clone:

```powershell
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe build_festival_voice.py --help
.\.venv\Scripts\python.exe run_tests.py
.\.venv\Scripts\python.exe run_gui.py
```

Close the GUI after the startup check and confirm no helper process was left
running. A migration is not complete until the complete suite and GUI startup
both succeed from the final checkout path.

## Checkout inside another repository

When this clone sits beneath another Git repository, the containing repository
should ignore the entire checkout path. If older versions of the files were
already tracked by the containing repository, remove them from that repository's
index only after making a backup and reviewing the exact staged boundary
change. An index-only removal must not delete the standalone working files.
