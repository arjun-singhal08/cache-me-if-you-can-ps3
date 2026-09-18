# ACV Team Git Workflow

Remember:

```bash
git pull
git status
git add <reviewed-files>
git commit -m "describe the ACV change"
git push
```

Before work:
```bash
git pull
git status
```

Before commit:
```bash
pytest tests/acv
git status
git add <reviewed-files>
git commit -m "Add ACV peer-relative baseline"
git push
```

Good messages:
- `Add ACV workbook schema parser`
- `Add ACV rank-decay scorer`
- `Add ACV peer residual features`
- `Add ACV ranking UI`
- `Fix leading-zero car IDs`

Do not blindly overwrite teammate conflicts.

Stage explicit reviewed paths. Keep datasets, credentials, local environments and unrelated teammate changes out of commits. Do not force-push shared branches.
