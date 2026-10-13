# CLAUDE.md

## Committing changes

When asked to commit, follow these steps in order:

1. Run `cargo fmt`.
2. Run `cargo clippy --all-targets -- -D warnings`.
   - If it reports warnings or errors, abort and ask the user to fix automatically.
3. Stage all changes, including new and deleted files, with `git add -A`.
4. Commit with a message that summarizes the changes:
   - First line: a concise summary of about 50 characters, in English, imperative mood
     (e.g. "Add BVH refit for deformed meshes").
   - If needed, add a blank line followed by bullet points explaining what changed and why.

## Pushing changes

**Do not push unless the user explicitly says "push". Exception: the "Version up" procedure may push a tag.**

When asked to push:

1. First execute the procedure in "Committing changes".
2. Run `git push`.


## Version up

1. Check if `cargo publish` is possible by looking at the dependencies in Cargo.toml files. The dependent crate should be published in crates io.
2. Commit the change following "committing changes", abort if there is any errors
3. Check GitHub and check the CI passes or not. If CI does not succeed, abort. 
4. Increment the version for "del-msh-cpu" and "del-msh-dlpack" and "del-msh-numpy" 
5. create the tag with the version name (e.g., "v0.1.50") and push the tag
6. Check GitHub and check the CI passes or not. If CI does not succeed, abort. 
7. Generate release note in mark down format such that the user can put it on GitHub. 