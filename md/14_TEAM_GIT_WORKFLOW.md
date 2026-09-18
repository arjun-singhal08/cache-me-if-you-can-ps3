# Team Git Workflow

## Mental model

- repository = shared project folder;
- commit = saved checkpoint;
- push = upload checkpoints;
- pull = download teammates' checkpoints;
- branch = isolated line of work;
- pull request = review/merge a branch into the main line.

## Before starting work

```bash
git switch main
git pull
```

For meaningful work create a branch:

```bash
git switch -c feature/shm-data-explorer
```

## During work

Check changes:

```bash
git status
```

Save a focused checkpoint:

```bash
git add <files>
git commit -m "Add SHM data profiling view"
```

Upload:

```bash
git push -u origin feature/shm-data-explorer
```

Then open a GitHub pull request into `main`.

## Team rules

- pull before starting;
- one responsibility per branch when possible;
- small commits with descriptive messages;
- do not commit raw official datasets;
- never commit `.env`, API keys, tokens, or passwords;
- do not force-push `main`;
- keep working model artifacts recoverable before risky experiments;
- resolve merge conflicts carefully rather than overwriting teammates' files.

## Suggested branches

- `feature/shm-data-explorer`
- `feature/shm-feature-engineering`
- `feature/shm-model`
- `feature/streamlit-ui`
- `chore/docker`
- `docs/shm-plan`
