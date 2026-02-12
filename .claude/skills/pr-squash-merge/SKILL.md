---
name: pr-squash-merge
description: Create a PR with a comprehensive description, then squash-merge it after user approval
disable-model-invocation: true
allowed-tools: Bash(git *), Bash(gh *)
---

# PR Squash-Merge Workflow

You are creating a pull request and squash-merging it. Follow these steps exactly.

## Step 1: Gather context

Run these commands to understand the branch:

- `git branch --show-current` — get current branch name
- `git log main..HEAD --oneline` — list commits on this branch
- `git log main..HEAD --format="%B---"` — get full commit messages
- `git diff main...HEAD --stat` — file change summary
- `git status` — ensure working tree is clean
- `gh pr list --head $(git branch --show-current)` — check for existing PR

If the working tree is dirty, warn the user and stop.
If a PR already exists, inform the user and ask how to proceed.

## Step 2: Draft the PR

Analyze ALL commits and the diff stat to write:

**Title:** A concise imperative summary (under 70 chars) covering the overall change, prefixed with conventional commit type (feat:, fix:, refactor:, etc.)

**Body:** Use this format:

```
## Summary
<3-5 bullet points covering the key changes across ALL commits>

## Commits squashed
<list each commit hash (short) and its subject line>

## Test plan
- [ ] <relevant verification steps>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## Step 3: Show the user the PR draft

Present the title and body to the user in a formatted preview. Then use AskUserQuestion to ask:

"Does this PR look good to create?"

Options:
- "Create PR" — proceed to create the PR
- "Edit title/description" — let user provide changes, then re-preview

Do NOT create the PR until the user approves.

## Step 4: Create the PR

Run: `gh pr create --title "<title>" --body "<body>"` using a HEREDOC for the body.

Show the user the PR URL.

## Step 5: Prompt for merge

Use AskUserQuestion to ask:

"PR created. Ready to squash-merge into main?"

Options:
- "Squash-merge now" — proceed
- "I'll merge later" — stop here

## Step 6: Squash-merge

If approved, run:

```
gh pr merge --squash --delete-branch
```

Then confirm the merge was successful and show the final state.
