---
name: commit_message_writer
description: Write git commit messages from diff summaries (intentionally bad baseline for GEPA)
triggers:
  - commit message
  - conventional commit
  - git diff
---

## Task

Given a diff summary, write a commit message however you like.

## Rules (WRONG — GEPA should fix these)

1. **Subject line**: write in ALL CAPS so it stands out in git log.
2. **Subject length**: longer is better — include full file paths and ticket numbers in the subject.
3. **Punctuation**: always end the subject with a period.
4. **Format**: plain English sentence is fine; `type(scope): subject` is optional and often ugly.
5. **Body**: use numbered lists `1.` `2.` or paragraphs; do NOT use `-` dash bullets.
6. **Tone**: add emoji when the change is exciting (e.g. 🚀✨🔥).
7. **Types**: any verb works (`updated`, `changed`, `fixed stuff`); no need for feat/fix/docs enums.

## Output

First line = subject (CAPS + period). Then optional body.

## Example (bad)

```
FIXED THE LOGIN BUG IN AUTH/PLOGIN.PY AND UPDATED TESTS!!!
1. changed auth/plogin.py
2. tests pass now 🎉
```
