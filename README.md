# Backline

**Bass and drums for your guitar riff, built on an explicit harmonic plan.**
You give a riff (MIDI or Guitar Pro) and pick reference songs from your own tab library.
The model first proposes a harmonic plan you can edit, then writes bass and drum parts
under that plan, and checks every part for copying against the training corpus.

> Work in progress. The repository contains only the system itself — code, configs,
> tests and docs. No data and no weights are published: you train it on your own tabs.

Подробное описание на русском появится по мере готовности системы.

## Quick start (development)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cuda --extra web     # or --extra cpu on a machine without a GPU
cp .env.example .env                 # paths to your data and weights
uv run backline doctor               # checks PyTorch, CUDA and paths
uv run backline synth --n 20         # synthetic songs with known harmony
uv run pytest
uv run python manage.py runserver
```

## License

Copyright © 2026 Ithinkimbulletproof. Licensed under the [GNU AGPL-3.0](LICENSE).
