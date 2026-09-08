# Installation and harness compatibility

Verified 2026-09-08 against official documentation, installer source and an isolated filesystem smoke test. Install only `skills/jax-pytorch-porting/`, not research archives or model dependencies. Read the package before granting execution permissions.

## Recommended: one shared local installation

With access to this private repository already configured through your Git credential helper:

```sh
npx --yes skills@1.5.25 add hcsolakoglu/jax-pytorch-porting-skill \
  --skill jax-pytorch-porting --global --agent codex claude-code cursor --yes
```

The pinned installer release was exercised with the local skill path in a temporary HOME. It created `~/.agents/skills/jax-pytorch-porting` and a Claude symlink under `~/.claude/skills/`; file hashes matched. Cursor and Codex use the shared directory, so separate `.cursor` or `.codex` copies were unnecessary. No actual user-global directory was modified. This verifies installation mechanics, not agent quality or every harness version. See [observed result](research/installation-smoke.json).

For private-repository authentication or network problems, clone once using existing credentials, then install from that checkout:

```sh
gh repo clone hcsolakoglu/jax-pytorch-porting-skill
cd jax-pytorch-porting-skill
npx --yes skills@1.5.25 add ./skills/jax-pytorch-porting \
  --skill jax-pytorch-porting --global --agent codex claude-code cursor --yes
```

Never place a token in a URL, command history or documentation. Installer updates can change paths; review changes before replacing an existing skill. The installer may replace an existing destination. Preserve local modifications and inspect duplicate names before an update. For project scope, omit `--global` and run from the intended project.

## Official discovery paths

Append `jax-pytorch-porting/SKILL.md` to each directory below. A local directory is not automatically synchronized to remote workers or another product.

| Harness | Project directory | User/global directory | Invoke or verify |
|---|---|---|---|
| Codex CLI / IDE | `.agents/skills/` | `~/.agents/skills/` | `/skills` or `$jax-pytorch-porting`; restart if discovery is stale |
| Claude Code | `.claude/skills/` | `~/.claude/skills/` | `/jax-pytorch-porting`; inspect enabled skills and permissions |
| Cursor | `.agents/skills/` or `.cursor/skills/` | `~/.agents/skills/` or `~/.cursor/skills/` | Customize → Skills; invoke through `/` |
| Antigravity | `.agents/skills/` | Product documentation differs; see below | Mention skill name and verify discovery in that product |
| Gemini CLI | `.agents/skills/` or `.gemini/skills/` | `~/.agents/skills/` or `~/.gemini/skills/` | `/skills list`, `/skills reload`; preserve activation consent |
| OpenCode | `.agents/skills/` or `.opencode/skills/` | `~/.agents/skills/` or `~/.config/opencode/skills/` | Native skill discovery, subject to skill permission |
| ChatGPT | Product-managed, not a local filesystem convention | Plugins → Skills | Create → Upload from your computer, then select the installed skill using `@` where supported |

Official sources: [Codex/ChatGPT authoring](https://learn.chatgpt.com/docs/build-skills), [Claude Code](https://code.claude.com/docs/en/skills), [Cursor](https://cursor.com/docs/skills), [Gemini CLI](https://geminicli.com/docs/cli/skills/), [OpenCode](https://opencode.ai/docs/skills), [ChatGPT upload and availability](https://help.openai.com/en/articles/20001066).

### Antigravity: do not guess a global path

Two current official pages disagree: [general documentation](https://antigravity.google/docs/skills/) lists `~/.gemini/config/skills/`, whereas [IDE documentation](https://antigravity.google/docs/ide/skills/) lists `~/.gemini/antigravity/skills/`. Both agree on project `.agents/skills/` and legacy `.agent/skills/` support. Prefer the common project directory until the exact installed product's global discovery is confirmed.

Our Vercel 1.5.25 smoke test included `--agent antigravity`: it installed only to shared `~/.agents/skills/`, not either documented Antigravity global path. An installer exit code of zero therefore does not prove global discovery by Antigravity. This is an observed path mismatch, not a claim that every Antigravity release fails to scan the shared directory. For a global manual installation, select the directory documented for your actual product, then verify that it appears in a new session.

Manual project install from the skill repository, after checking the destination does not exist:

```sh
# Replace /path/to/model-project with your intended project, not the skill repo.
destination=/path/to/model-project/.agents/skills/jax-pytorch-porting
mkdir -p "$(dirname "$destination")"
test ! -e "$destination" && test ! -L "$destination" && \
  cp -R skills/jax-pytorch-porting "$destination"
```

For Claude's manual installation, use `.claude/skills/` instead. Use the matching global directory from the table for a manual user installation. Do not copy only `SKILL.md`: references and the parity helper are part of the bundle.

### ChatGPT and remote environments

Upload `dist/skill.zip` through Plugins → Skills → Create → Upload from your computer where your account/workspace supports uploaded skills. Availability and permission differ by plan, workspace and surface. Upload is scanned and may require review. Installing into local Codex directories does not install a skill into ChatGPT web. This project did not perform a ChatGPT UI upload or change workspace settings.

Cursor's official docs distinguish local `~/.agents/skills/` from optional Cloud Agent syncing of `~/.cursor/skills/`. Use repository-scoped skills or approved worker-image installation for remote execution rather than assuming local home-directory files are present. Do not enable cloud syncing or publish this private repository without authorization.

## Portability contract

Core instructions use only standard name/description YAML and relative references. They do not pin an agent model, spawn subagents, prescribe a provider, auto-install an accelerator stack or grant permissions. `agents/openai.yaml` supplies optional OpenAI UI metadata; other hosts may ignore it. Agent tool discovery, shell permissions, supported execution locations and confirmation requirements remain host-owned.

Source inspection: [Vercel installer](https://github.com/vercel-labs/skills/blob/7ffbeb96f012a63c0583a2e71e24385dc497566d/src/installer.ts) resolves universal agents to a canonical path before consulting legacy per-agent global path fields; [agent definitions](https://github.com/vercel-labs/skills/blob/7ffbeb96f012a63c0583a2e71e24385dc497566d/src/agents.ts) alone are insufficient to predict installation. The smoke test uses the published npm release, not an assumption that Git main equals that release.

## Bounded installation test

`python tools/install_smoke.py` uses an isolated temporary HOME, disables telemetry, closes stdin, bounds subprocess runtime and removes its temporary files. Closing stdin resolved an initial timed-out job-control stall in this environment. It invokes no paid agent or model. Do not generalize this local diagnosis into a service-outage claim.
