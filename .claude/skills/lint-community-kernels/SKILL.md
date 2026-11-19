---
name: lint-community-kernels
description: Run ROCm/rocm-community-kernels' lint checks exactly as CI's pre-commit job does (black, mdformat, actionlint, whitespace, tabs, YAML/JSON, large files, *_test.py naming), including new untracked files, and fix what is mechanically fixable. Use before every commit in this repository and whenever the "pre-commit / Run hooks" check fails.
---

# Lint the repository

CI's [`pre-commit.yml`](../../../.github/workflows/pre-commit.yml) runs the hooks in
[`.pre-commit-config.yaml`](../../../.pre-commit-config.yaml) on every file in the
pull request's checkout, with Python 3.12. Its job, "Run hooks", is a required check: a
pull request can't merge while it fails, and it fails if any hook reports a problem or
changes a file.

## Run it

Use pre-commit itself, never `black` or `mdformat` from the system: the hook versions
are pinned in `.pre-commit-config.yaml`, and other versions format differently. Keep
the virtual environment outside the repository, or the hooks will lint it too.

```bash
python3 -m venv ~/.venvs/pre-commit && ~/.venvs/pre-commit/bin/pip install pre-commit   # once
~/.venvs/pre-commit/bin/pre-commit run --all-files                                      # what CI runs
```

`--all-files` only covers files git tracks. New files that aren't committed or staged
yet are skipped, though CI will check them once they're in the pull request. To include
them without staging anything:

```bash
~/.venvs/pre-commit/bin/pre-commit run --files $(git ls-files) $(git ls-files --others --exclude-standard)
```

The first run downloads the hooks from GitHub (cached in `~/.cache/pre-commit`). If a
hook can't be installed, run the others with `SKIP=<hook id>` and tell the user which
hook didn't run; CI will still run it.

## The hooks

| Hook                                                            | Checks                                                                                                                                                | Fixes files? |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending` | Whitespace at line ends, one newline at the end of files, consistent line endings                                                                     | Yes          |
| `check-yaml`, `check-json`                                      | YAML and JSON parse                                                                                                                                   | No           |
| `check-merge-conflict`                                          | No leftover conflict markers                                                                                                                          | No           |
| `check-added-large-files`                                       | No file over 500 kB among files staged as new. CI stages nothing, so it doesn't catch them there                                                      | No           |
| `requirements-txt-fixer`                                        | Sorted `requirements*.txt`                                                                                                                            | Yes          |
| `black`                                                         | Python formatting (25.11.0)                                                                                                                           | Yes          |
| `mdformat`                                                      | Markdown formatting, with GFM tables, front matter and black for Python code blocks. Skips `CARD.md` and the card template, which are Jinja templates | Yes          |
| `forbid-tabs`                                                   | No tab characters, in any file type, including `.hip`, `.cpp` and `.h`                                                                                | No           |
| `actionlint`                                                    | GitHub workflow files (shellcheck off; the self-hosted `gfx950` label is allowed)                                                                     | No           |
| `test-file-naming`                                              | Test files under `tests/` are named `*_test.py`, not `test_*.py`                                                                                      | No           |

## Fix and report

1. Run the hooks. Fixers have already rewritten files when they report "Failed" with
   "files were modified by this hook"; run again to confirm they pass.
1. Fix the report-only failures by hand:
   - **forbid-tabs**: replace tabs with spaces (`expand -t 4 file > tmp && mv tmp file`
     keeps alignment). Kernel sources copied from elsewhere often use tabs.
   - **test-file-naming**: `git mv tests/test_x.py tests/x_test.py`. Kernels ported
     from elsewhere often use `test_*.py`.
   - **check-added-large-files**: don't commit build outputs (`build/`, `result*`,
     `*.so`). A source file over 500 kB, such as generated instances, needs a maintainer
     to agree before it goes in. Look for them with `find <kernel> -size +500k`.
   - **actionlint**, **check-yaml**, **check-json**: fix the line it reports.
1. Show `git diff --stat`, and point out any file that was clean before and is now
   changed by a fixer: committing it would mix unrelated formatting into the change.

The pull request CI's other static checks (how kernels are wrapped, the community rules,
the maintainers registry) are in the `check-community-kernel` skill.

## Rules

- Never stage, commit or push. Leave fixes in the working tree for the user to review.
- Report honestly: if a hook fails and you can't fix it, or didn't run, say so with its
  output.
