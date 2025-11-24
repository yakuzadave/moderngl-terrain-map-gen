---
applyTo: '**'
---

# Changelog guidelines

These instructions specialize GitHub Copilot for maintaining the project **changelog**.

The goal is to keep a clear, human friendly history of changes that matches the codebase and is easy to scan by version.

---

## Location and naming

- The canonical changelog file is:

  - `CHANGELOG.md` at the **repository root**.

- The root `README.md` should link to it as:

  - `[Changelog](./CHANGELOG.md)`

AI should assume this file exists or should be created if it does not.

---

## Format and structure

Use a Markdown format inspired by **Keep a Changelog** and **Semantic Versioning**.

Basic structure:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Removed
- ...

### Deprecated
- ...

### Security
- ...

## [1.2.0] - 2025-11-23

### Added
- ...

### Changed
- ...
````

Rules:

* Top level title: `# Changelog`.
* One `## [Unreleased]` section at the top.
* One `## [x.y.z] - YYYY-MM-DD` section per released version.
* Inside each version, use these subsection headings as needed:

  * `### Added`   (new features)
  * `### Changed` (behavior changes that are not breaking)
  * `### Fixed`   (bug fixes)
  * `### Removed` (features removed)
  * `### Deprecated` (features to be removed later)
  * `### Security` (security related fixes or changes)

Subsections that are empty can be omitted.

---

## Entry style

Each entry should be:

* A single bullet point.
* Written in **concise, descriptive language**.
* Past tense or descriptive present is fine, but be consistent.

Examples:

```markdown
### Added
- Added configurable biome presets for coastline, mountains, and marsh.
- Added headless ModernGL render path for batch terrain baking.

### Changed
- Updated default heightmap resolution to 1024x1024 for test renders.
- Switched Pillow image saves to PNG for lossless debug outputs.

### Fixed
- Fixed inverted normal maps in terrain preview renders.
- Fixed crash when loading configs without explicit biome thresholds.
```

Avoid:

* Internal shorthand that future readers will not understand.
* Referring to specific commit hashes unless necessary.
* Rewriting history of past releases except to fix typos or obvious mistakes.

---

## Versioning rules

Assume **Semantic Versioning**:

* `MAJOR.MINOR.PATCH` (for example `2.1.3`)

Guidance:

* Increment **MAJOR** when:

  * You introduce breaking changes to public APIs, config schemas, or CLI behavior.

* Increment **MINOR** when:

  * You add backwards compatible features or significant enhancements.

* Increment **PATCH** when:

  * You fix bugs or make small internal improvements that do not affect public APIs.

---

## When and how to update the changelog

AI should keep the changelog updated alongside meaningful code changes.

For each pull request or major local change:

1. Identify the type of change:

   * New feature
   * Behavior change
   * Bug fix
   * Removal or deprecation
   * Security fix

2. Add an entry under `## [Unreleased]` in the appropriate subsection.

3. Use short but specific phrasing:

   * Mention modules or features by name.
   * Mention user visible behavior when relevant.

When preparing a release:

1. Change the current `## [Unreleased]` content into a versioned section:

   ```markdown
   ## [1.3.0] - 2025-11-23
   ```

2. Add a fresh empty `## [Unreleased]` section above it with all the standard headings to be used for future work.

3. Do not move already released entries into another version unless correcting a mistake.

---

## Cross references and links

Where helpful, entries can refer to:

* Issue or ticket numbers: `(#123)`
* Pull requests: `(PR #456)`
* External docs: short links

Example:

```markdown
### Fixed
- Fixed crash when saving configs without lighting settings (#132).
```

Keep references short and at the end of the line.

---

## How AI should behave with the changelog

When generating or modifying code that:

* Adds a new feature,
* Changes behavior,
* Fixes a bug,
* Alters configuration or public APIs,

AI should:

1. Locate `CHANGELOG.md`.
2. Add or update a bullet under the `## [Unreleased]` section and the appropriate category.
3. Use the existing style and phrasing used in the file.

When reviewing or refactoring:

* Ensure new behavior described in code matches the changelog.
* Suggest adding or adjusting entries when notable changes lack documentation.

When editing `CHANGELOG.md`:

* Preserve the overall structure.
* Do not reorder historical versions.
* Avoid removing historical entries except for obvious typos or duplication.
