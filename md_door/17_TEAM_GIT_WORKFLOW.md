# Door Team Git Workflow

Use small checkpoints.

Before work:

```bash
git pull
```

Check changes:

```bash
git status
```

Save work:

```bash
git add <specific-files>
git commit -m "door: improve cycle segmentation"
git push
```

## Suggested commit prefixes

- `door: data ...`
- `door: segment ...`
- `door: model ...`
- `door: ui ...`
- `door: test ...`
- `door: docs ...`

## Team safety

- Pull before editing shared files.
- Avoid everyone editing `app.py` simultaneously.
- Keep segmentation/classifier code in Door-specific modules.
- Never commit API keys or personal absolute dataset paths.
- Do not force-push shared branches during the hackathon unless the team explicitly coordinates it.
