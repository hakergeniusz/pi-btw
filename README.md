# pi-btw

`/btw` -- ask a side question without interrupting the main conversation.

## How it works

- Extension commands in pi are dispatched *before* the streaming check, so `/btw` runs
  immediately even while the main agent is mid-turn. Nothing is queued, nothing is
  aborted.
- The side question reuses the exact system prompt, message prefix and tool declarations
  the main thread is already sending, so the request hits the provider's prompt cache
  instead of re-paying for the whole conversation.
- The answer is rendered in an overlay and is never appended to the session, so it costs
  zero main-context tokens.
- The answer streams in as it arrives. Esc / Enter / Space dismiss it and abort an
  in-flight request; Up / Down (or Ctrl+P / Ctrl+N) scroll long answers.

An independent implementation of the same idea as Claude Code's `src/commands/btw` and
`src/utils/sideQuestion.ts`.

## Install

```bash
pi install git:github.com/hakergeniusz/pi-btw
```

MIT licensed.
