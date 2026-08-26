# Audio and voice tests — 2026-08-16

Build under test: `stage-default-v8-controller-stop-mango45` and
`stage-default-v9-idle-inference-gate` (voice matching is identical in both).

Recorded after the fact from the session transcript. Each row is a real call
against the device, not a simulation.

## Voice commands (live, DJI MIC MINI, hw:0,0)

Wake phrase `hey wendy`, `ACTION_MODE=border_collie`, actions armed.

| transcript | matched | input level | audio | result |
|---|---|---|---|---|
| `Go to the mango` | yes | **-55.6 dBFS** | 2140 ms | started run `fc9dca84` |
| `can you go find the mango?` | yes | **-54.2 dBFS** | 3180 ms | matched |
| `Can you go to the mango?` | yes | **-54.8 dBFS** | 1980 ms | matched |

All three matched through the legacy verb+fruit rule (a motion verb plus
exactly one fruit), and all three started real Demo Runs.

**The finding is the level, not the matching.** Every utterance landed at
about -54 to -55 dBFS, roughly 25 dB below where close-mic speech should sit
(-20 to -30). That is the regime where a quiet two-syllable word degrades in
transcription, which is what the operator experienced as intermittent
"invalid command" while saying mango. Raising the microphone gain or speaking
closer is the primary fix; the fuzzy matcher added afterwards only salvages
transcriptions that already degraded.

No failing transcript was captured live. The three above are the only
recorded samples, and all succeeded.

## Fuzzy mango matcher (offline, against the deployed matcher)

25 probes run directly against `interpret_command`. Accepted:

`mango` · `mengo` · `man go` · `go to the mengo` ·
`can you go find the mangoe` · `hey wendy mango please`

Rejected (no action):

`tango` · `bingo` · `bango` · `banjo` · `dingo` · `man` · `many` · `mangle` ·
`manage` · `monkey` · `go and read the manga` · `go look at that mangy dog` ·
`how many are there go ahead` · `the man goes over there` ·
`can you go and manage the queue` · `run the full demo and find the mengo` ·
`stopp`

`stop` and `full demo` resolved correctly and are evaluated before any mango
canonicalisation, so fuzzy matching cannot reach across them.

## Bark and speaker

| call | result |
|---|---|
| `POST :8111/api/bark` (raw sidecar) | `{"ok": true, "uuid": "161387de-…"}` |
| `POST :8110/api/thermal/beep` (first, through the policy) | `sounded: false` — `speaker remute failed: speaker volume verification expected 0, got 6` |
| `POST :8110/api/bark/test` (after the settle fix) | `sounded: true`, `speaker_remuted: true`, trace `audible@6 → bark_requested → muted@0` |

**Operator reported silence in every case, including the clean cycle.**

So volume control is proven working — the VUI connected, set volume 6, and
read it back — while nothing was audible. That points past the mute policy to
the audio itself: `play_by_uuid` discards its response, the sidecar returned
`ok` unconditionally, and `bark_ready` only checked the uuid was *configured*,
never that it exists on this robot. Commit `b8d0357` adds
`bark_uuid_present` / `audio_uuids` so the next reading of `:8111/status`
answers it directly. Not yet deployed.

The first failure was also a real bug: `_set_and_verify` read `GetVolume`
immediately after `SetVolume` and the Go2 returns the previous level for a
moment. That latched the policy not-ready, which would have refused every
later bark and thermal alarm until restart.

## Thermal beep

`WOOF_THERMAL_BEEP_URL` pointed at `http://127.0.0.1:8110/api/thermal/beep`,
which did not exist — both `:8110` and `:8111` returned 404. Recorded during
the 82 C motor critical at 17:05:35:

```
alert=critical  beep_requested=true  beep_ok=false
error: HTTPError: HTTP Error 404: Not Found
```

The alarm has never sounded. The endpoint now exists. Separately, every
`warning` sample shows `beep_requested=false` — only `critical` requests a
beep, so the rise-rate warning that fired twelve minutes before the shutdown
would still be silent. That change lives in the thermal-monitor deployment,
not this repo.
