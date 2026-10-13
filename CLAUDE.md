# CLAUDE.md

## Committing changes

When asked to commit, follow these steps in order:

1. Run `cargo fmt`.
2. Run `cargo clippy --all-targets -- -D warnings`.
   - If it reports warnings or errors, fix them, then start again from step 1.
   - If you cannot fix them yourself, do not commit. Report the problem and wait for instructions.
3. Stage all changes, including new and deleted files, with `git add -A`.
4. Commit with a message that summarizes the changes:
   - First line: a concise summary of about 50 characters, in English, imperative mood
     (e.g. "Add BVH refit for deformed meshes").
   - If needed, add a blank line followed by bullet points explaining what changed and why.
