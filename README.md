# FestVox Speech GUI

FestVox Speech GUI is an independent Windows editor and voice-building toolkit
for Festival and FestVox. It is not the FestVox project itself. It converts
UTAU recordings and `oto.ini` timing into generated Festival and UniSyn voices,
then lets you synthesize and edit speech through a PyQt GUI.

The primary runtime is Festival inside WSL. English, Asaxi, and Japanese use
separate linguistic frontends while sharing the same waveform, timing, pitch,
voicing, join, project, and export tools.

This repository contains code and metadata only. It does not include UTAU
voicebanks or generated voices.

## What it provides

- A Windows GUI for speech generation, waveform timing, pitch, voicing,
  vocal-tract, mora/accent, unit-choice, join, and sentence-project editing.
- A guarded builder that creates real Festival/UniSyn voices without writing
  into the source UTAU bank.
- English ARPAsing, Asaxi over ARPAsing, and Japanese CV, VCV, and CVVC paths.
- Optional integrated ARPAsing voices that expose English, Asaxi, and Japanese
  frontends over one recording database.
- Context-sensitive alternative-unit selection with per-occurrence manual
  overrides.
- A smaller pure-Python diphone renderer for compatibility and diagnostics.

## Supported voice routes

| Language | Source bank | Builder selection |
| --- | --- | --- |
| English | ARPAsing | `--language en --bank-type arpasing` |
| Asaxi | ARPAsing with the required inventory | `--language asaxi --bank-type arpasing` |
| Japanese | Japanese CV, VCV, or CVVC | `--language ja --bank-type cv`, `vcv`, or `cvvc` |
| English + Asaxi + Japanese | Compatible ARPAsing bank | English primary plus two `--enable-language` options |

Japanese-only banks stay in a Japanese alias namespace. An integrated
ARPAsing build uses the bundled phoneme profile to route Japanese phones into
the shared inventory; it does not reinterpret the English ARPAsing mapping.

## Requirements

- Windows 10 or 11.
- Python 3 with `pip`.
- WSL with an Ubuntu distribution for the main Festival engine.
- A legally obtained UTAU/OpenUtau bank when building a voice.
- One explicitly selected pitch/OTO scope per generated voice.

Inside Ubuntu, install Festival, Speech Tools, the built-in Kal test voice,
and the development libraries used by the project-local crossover runtime:

```bash
sudo apt update
sudo apt install festival festvox-kallpc16k speech-tools \
  g++ festival-dev libestools-dev libsystemd-dev libncurses-dev
```

If WSL is not installed yet, run this once in an administrator PowerShell and
restart when Windows asks:

```powershell
wsl --install -d Ubuntu
```

## Quick start

Clone the repository and create a local virtual environment:

```powershell
git clone https://github.com/wik-wav/festvox-speech-gui.git
cd festvox-speech-gui

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

Build the project-local Festival helper after installing the WSL packages:

```powershell
.\.venv\Scripts\python.exe `
  src\festvox_tts\native_unisyn\build_wsl_runtime.py `
  --distro Ubuntu
```

Launch the editor:

```powershell
.\.venv\Scripts\python.exe run_gui.py
```

On first launch:

1. Open **Options > WSL / Festival settings**.
2. Confirm the WSL distribution and Festival binary, then use **Test
   connection**.
3. Select **Festival via WSL**, **English**, and `kal_diphone` to verify the
   installation before building another voice.
4. Enter text and choose **Generate Audio**.

The normal generated-voice location is `generated_voices/` in this checkout.
You can choose another Windows folder under **Options > WSL / Festival
settings** and refresh it with the Voicebank Manager.

## Build a voice

`build_festival_voice.py` is the stable Windows entry point. Use Windows paths;
the builder derives WSL paths when it calls Festival and EST.

Important safety and scope rules:

- `--samples` is read only.
- `--output` must not be inside the source bank.
- Select exactly one pitch using one `oto.ini` file or one single-pitch folder.
- A rebuild into an existing generated folder requires `--overwrite`.
- `--test` loads the generated Festival entry point and creates a smoke-test
  WAV beneath the generated output.

### English ARPAsing

```powershell
$Python = ".\.venv\Scripts\python.exe"
$Source = "C:\UTAU\voice\MyEnglishBank\F3"
$Output = Join-Path $PWD "generated_voices\my_english_voice"

& $Python .\build_festival_voice.py `
  --language en `
  --bank-type arpasing `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output $Output `
  --name my_english_voice `
  --test `
  --test-text "this is a test"
```

### Japanese CVVC

Change `cvvc` to `cv` or `vcv` only when that is the intended bank type.

```powershell
$Python = ".\.venv\Scripts\python.exe"
$Source = "C:\UTAU\voice\MyJapaneseBank\F3"
$Output = Join-Path $PWD "generated_voices\my_japanese_voice"

& $Python .\build_festival_voice.py `
  --language ja `
  --bank-type cvvc `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output $Output `
  --name my_japanese_voice `
  --test
```

### Integrated English, Asaxi, and Japanese

Use this only when the selected ARPAsing bank contains the additional phones
needed by the enabled frontends.

```powershell
$Python = ".\.venv\Scripts\python.exe"
$Source = "C:\UTAU\voice\MyIntegratedBank\E3"
$Output = Join-Path $PWD "generated_voices\my_integrated_voice"

& $Python .\build_festival_voice.py `
  --language en `
  --enable-language asaxi `
  --enable-language ja `
  --bank-type arpasing `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output $Output `
  --name my_integrated_voice `
  --test `
  --test-text "this is a test"
```

For an Asaxi-only configuration, use `--language asaxi --bank-type arpasing`.

The Japanese-only `--profile` option accepts a bank-analysis profile JSON with
explicit alias classifications and allophone settings. It is not a required
pitch setting. Integrated ARPAsing builds instead use the bundled
`src/festvox_tts/profiles/en-jap-mapping.yaml`; override that mapping only with
`--phoneme-map`.

Run the primary builder help at any time:

```powershell
.\.venv\Scripts\python.exe build_festival_voice.py --help
```

The full option reference, source-window behavior, voice colors, OTO metadata,
and output layout are documented in
[Unified Voice Builder](docs/UNIFIED_VOICE_BUILDER.md).

## Use a generated voice

When the output is under the configured generated-voice root, the GUI discovers
it automatically. Otherwise:

1. Open **Voicebank > Voicebank manager**.
2. Add or refresh the folder containing the generated voice.
3. Select a language supported by the voice manifest.
4. Select the voice and generate once before editing timing or parameters.

The Voicebank Manager can permanently delete generated voices after a strong
confirmation. It refuses built-in Festival voices and source UTAU folders.

## Optional dependencies

Dependency-free kana/romaji Japanese analysis is always available. Install the
Open JTalk frontend for kanji morphology and full-context labels:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-japanese-optional.txt
```

Install the shared FRQ-less source-F0 fallback when a bank has no usable UTAU
frequency data:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-source-f0.txt
```

`sounddevice` improves local playback and `librosa` enables an optional
higher-quality time-stretch path. The GUI has fallbacks when they are absent.

## Pure-Python compatibility renderer

The Festival/WSL engine is the main application path. For a smaller diagnostic
database that does not require Festival:

```powershell
.\.venv\Scripts\python.exe src\festvox_tts\utau2festvox.py `
  --bank "C:\UTAU\voice\MyEnglishBank\F3" `
  --out ".\generated_voices\my_python_db" `
  --name my_python_db

.\.venv\Scripts\python.exe src\festvox_tts\synth_diphone.py `
  "this is a test" `
  --lang en `
  --db ".\generated_voices\my_python_db" `
  --out ".\rendered_audio\test.wav"
```

Copy `festvox.example.json` to the ignored `festvox.json` only if you want
named databases and persistent pure-Python defaults.

## Troubleshooting

- **`wsl.exe` is unavailable:** install WSL or select its executable in the
  Festival settings.
- **Festival is unavailable inside WSL:** install the Ubuntu packages above,
  then rerun `check_environment.py` and **Test connection**.
- **The builder reports multiple pitches:** point `--samples` and `--oto` at
  one pitch/subbank. Build another generated voice for another pitch.
- **A generated voice is missing from the GUI:** make sure its parent is the
  configured Windows generated-voice root, then refresh the Voicebank Manager.
- **The first render is slow:** a cold WSL/Festival process must start and load
  the voice; later renders use the warm helper.
- **A build folder already exists:** verify that it is the intended generated
  output, then rerun with `--overwrite`. Never use a source bank as output.

For a machine-readable installation report:

```powershell
.\.venv\Scripts\python.exe check_environment.py --json
```

## Documentation

- [Documentation map](docs/README.md)
- [Standalone repository workflow](docs/STANDALONE_REPOSITORY_WORKFLOW.md)
- [Complete UTAU-to-synthesis walkthrough](docs/GUIDE.md)
- [Unified Festival voice builder](docs/UNIFIED_VOICE_BUILDER.md)
- [GUI manual](src/festvox_tts/festvox_gui/README.md)
- [Join synthesis](docs/JOIN_SYNTHESIS.md)
- [Japanese integration design](docs/JAPANESE_UTAU_INTEGRATION_DESIGN.md)
- [Asaxi pitch integration](docs/ASAXI_PITCH_INTEGRATION.md)

## Development

Run the complete test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

Repository structure:

- `src/festvox_tts/`: synthesis, language frontends, builder, GUI, profiles,
  diagnostics, and native runtime source.
- `tests/`: complete core and headless GUI regression suites.
- `docs/`: operating guides, architecture notes, and implementation reports.
- `generated_voices/`, `rendered_audio/`, caches, and local configuration:
  ignored local output.

Maintainer and coding-agent priorities are retained in [AGENTS.md](AGENTS.md).
The former implementation-heavy root README is preserved as
[Development and implementation notes](docs/DEVELOPMENT_AND_IMPLEMENTATION_NOTES.md).
