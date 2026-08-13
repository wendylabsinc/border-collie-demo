"""Small stateless website/API for publishing operator commands into A."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol
from uuid import UUID

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from robotkit.client import WorldStateClient
from robotkit.contracts import (
    EffectRecord,
    EventRecord,
    GoalRecord,
    Observation,
    PublishResult,
    WorldSnapshot,
)
from robotkit.perception.website.diagnostics import build_debug_snapshot
from robotkit.perception.website.producer import WebsiteCommandProducer
from robotkit.fruits import SUPPORTED_FRUITS
from robotkit.runtime import deployment_generation, instance_id, world_state_url


class ObservationPublisher(Protocol):
    def publish_observation(self, observation: Observation) -> PublishResult: ...


class StateReader(Protocol):
    def snapshot(self) -> WorldSnapshot: ...

    def current_goal(self) -> GoalRecord | None: ...

    def latest_effect(self, goal_id: UUID | None = None) -> EffectRecord | None: ...

    def events(self, *, after: int = 0, limit: int = 100) -> list[EventRecord]: ...


class WebsiteCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, max_length=500)
    request_id: UUID | None = None


class WebsiteCommandReceipt(BaseModel):
    event_id: UUID
    revision: int
    duplicate: bool
    stream: str
    intent: str
    slots: dict[str, object]


def create_app(
    publisher: ObservationPublisher | None = None,
    *,
    producer: WebsiteCommandProducer | None = None,
    state_reader: StateReader | None = None,
) -> FastAPI:
    client = publisher or WorldStateClient(world_state_url())
    owns_client = publisher is None
    reader = state_reader or (client if publisher is None else None)
    command_producer = producer or WebsiteCommandProducer(
        producer_id=os.getenv("ROBOTKIT_PRODUCER_ID", "go2-website-command"),
        instance_id=instance_id(),
        deployment_generation=deployment_generation(),
        intent_ttl_seconds=float(os.getenv("WEB_INTENT_TTL_SECONDS", "30")),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_client:
            client.close()  # type: ignore[attr-defined]

    app = FastAPI(title="RobotKit Command Website", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/debug", response_class=HTMLResponse)
    def debug_page() -> str:
        return _DEBUG_PAGE

    @app.get("/v1/debug")
    def debug_state() -> dict[str, object]:
        if reader is None:
            raise HTTPException(status_code=503, detail="world-state reader unavailable")
        snapshot = reader.snapshot()
        goal = reader.current_goal()
        effect = reader.latest_effect(goal.goal_id if goal is not None else None)
        events = reader.events(after=max(0, snapshot.revision - 100), limit=100)
        supported_targets = tuple(
            value.strip()
            for value in os.getenv(
                "ROBOTKIT_MISSION_TARGETS", ",".join(sorted(SUPPORTED_FRUITS))
            ).split(",")
            if value.strip()
        )
        return build_debug_snapshot(
            snapshot,
            current_goal=goal,
            latest_effect=effect,
            events=events,
            supported_targets=supported_targets,
        )

    @app.post(
        "/v1/command",
        response_model=WebsiteCommandReceipt,
        status_code=status.HTTP_201_CREATED,
    )
    def publish_command(command: WebsiteCommand) -> WebsiteCommandReceipt:
        observation = command_producer.observation_from_command(
            command.command,
            request_id=command.request_id,
        )
        result = client.publish_observation(observation)
        return WebsiteCommandReceipt(
            event_id=observation.event_id,
            revision=result.revision,
            duplicate=result.duplicate,
            stream=observation.stream,
            intent=str(observation.payload["intent"]),
            slots=observation.payload["slots"],
        )

    return app


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RobotKit command</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: #101417; color: #edf3f5; }
    main { width: min(42rem, calc(100% - 2rem)); padding: 2rem; border: 1px solid #344047;
      border-radius: 1rem; background: #182025; box-sizing: border-box; }
    h1 { margin-top: 0; } label { display: block; margin: 1rem 0 .35rem; color: #b9c8ce; }
    input, button { font: inherit; border-radius: .5rem; border: 1px solid #46565e;
      padding: .75rem; box-sizing: border-box; }
    input { width: 100%; color: inherit; background: #0f1518; }
    button { margin-top: 1rem; color: #07120d; background: #64dc9b; border: 0; cursor: pointer; }
    button:disabled { opacity: .5; cursor: wait; }
    #status { min-height: 1.5rem; margin-top: 1rem; color: #b9c8ce; }
  </style>
</head>
<body><main>
  <h1>RobotKit command</h1>
  <p>Send an auditable instruction through the website perception channel.</p>
  <p><a href="/debug">Open live pipeline diagnostics</a></p>
  <form id="command-form">
    <label for="command">Instruction</label>
    <input id="command" value="find apple" maxlength="500" required autofocus>
    <button id="send" type="submit">Send instruction</button>
  </form>
  <div id="status" role="status" aria-live="polite"></div>
</main><script>
const form = document.getElementById('command-form');
const statusBox = document.getElementById('status');
const button = document.getElementById('send');
form.addEventListener('submit', async (event) => {
  event.preventDefault(); button.disabled = true; statusBox.textContent = 'Publishing…';
  try {
    const response = await fetch('/v1/command', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({command: document.getElementById('command').value})
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    statusBox.textContent = `Accepted ${body.intent} at world revision ${body.revision}.`;
  } catch (error) { statusBox.textContent = `Not sent: ${error.message}`; }
  finally { button.disabled = false; }
});
</script></body></html>"""


_DEBUG_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RobotKit diagnostics</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif;
      --bg:#0d1215; --panel:#172025; --line:#344047; --muted:#a9bbc2;
      --ok:#64dc9b; --waiting:#77bdfb; --warning:#ffd166; --stale:#ff9f43; --error:#ff6b6b; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:#edf3f5; }
    main { width:min(88rem,calc(100% - 2rem)); margin:auto; padding:1.5rem 0 3rem; }
    header { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; }
    a { color:#8ed0ff; } .muted { color:var(--muted); }
    #pipeline { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.75rem; margin:1.25rem 0; }
    .card, section { border:1px solid var(--line); border-radius:.8rem; background:var(--panel); padding:1rem; }
    .card { border-top:4px solid var(--waiting); min-height:10rem; }
    .card.ok { border-top-color:var(--ok); } .card.waiting { border-top-color:var(--waiting); }
    .card.warning { border-top-color:var(--warning); } .card.stale { border-top-color:var(--stale); }
    .card.error { border-top-color:var(--error); }
    .badge { text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; color:var(--muted); }
    h1,h2,h3 { margin-top:0; } h3 { text-transform:capitalize; margin-bottom:.5rem; }
    details { margin-top:.75rem; } pre { white-space:pre-wrap; overflow-wrap:anywhere; color:#c8d5da; font-size:.75rem; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
    table { width:100%; border-collapse:collapse; font-size:.82rem; }
    th,td { text-align:left; padding:.45rem; border-bottom:1px solid #29343a; vertical-align:top; }
    .event { padding:.6rem 0; border-bottom:1px solid #29343a; }
    .event:last-child { border:0; } #error { color:var(--error); min-height:1.4rem; }
    @media (max-width:72rem) { #pipeline { grid-template-columns:repeat(2,1fr); } }
    @media (max-width:48rem) { #pipeline,.grid { grid-template-columns:1fr; } }
  </style>
</head>
<body><main>
  <header><div><h1>RobotKit live diagnostics</h1>
    <div id="meta" class="muted">Connecting to A…</div></div>
    <a href="/">Send a command</a></header>
  <div id="error" role="alert"></div>
  <div id="pipeline" aria-live="polite"></div>
  <div class="grid">
    <section><h2>World observations</h2><div style="overflow:auto"><table>
      <thead><tr><th>Stream</th><th>Age</th><th>TTL</th><th>Revision</th><th>Frame</th></tr></thead>
      <tbody id="observations"></tbody></table></div></section>
    <section><h2>Black-box timeline</h2><div id="timeline"></div></section>
  </div>
</main><script>
const pipeline = document.getElementById('pipeline');
const observations = document.getElementById('observations');
const timeline = document.getElementById('timeline');
const meta = document.getElementById('meta');
const errorBox = document.getElementById('error');
const allowedStatus = new Set(['ok','waiting','warning','stale','error']);
function node(tag, text, className) {
  const item = document.createElement(tag); if (text !== undefined) item.textContent = text;
  if (className) item.className = className; return item;
}
function render(data) {
  errorBox.textContent = '';
  meta.textContent = `A revision ${data.revision} · state ${data.state_revision} · ${data.captured_at} · overall ${data.overall_status}`;
  const openStages = new Set([...pipeline.querySelectorAll('article[data-stage] details[open]')]
    .map(details => details.parentElement.dataset.stage));
  pipeline.replaceChildren();
  for (const stage of data.stages) {
    const status = allowedStatus.has(stage.status) ? stage.status : 'warning';
    const card = node('article', undefined, `card ${status}`);
    card.dataset.stage = stage.name;
    card.append(node('div', status, 'badge'), node('h3', stage.name), node('div', stage.summary));
    if (stage.age_seconds !== null) card.append(node('div', `Age ${stage.age_seconds}s`, 'muted'));
    const details = node('details'); details.open = openStages.has(stage.name);
    details.append(node('summary','Details'), node('pre',JSON.stringify(stage.details,null,2)));
    card.append(details); pipeline.append(card);
  }
  observations.replaceChildren();
  for (const observation of data.observations) {
    const row = node('tr');
    for (const value of [observation.stream, `${observation.age_seconds}s${observation.stale?' stale':''}`,
      `${observation.ttl_seconds}s`, observation.revision, observation.frame_id || '—']) row.append(node('td',String(value)));
    observations.append(row);
  }
  timeline.replaceChildren();
  for (const event of data.timeline) {
    const item = node('div',undefined,'event');
    item.append(node('strong',`r${event.revision} ${event.category}`),
      node('div',event.recorded_at,'muted'),node('pre',JSON.stringify(event.summary,null,2)));
    timeline.append(item);
  }
}
async function refresh() {
  try { const response = await fetch('/v1/debug',{cache:'no-store'}); const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`); render(body);
  } catch (error) { errorBox.textContent = `Diagnostics unavailable: ${error.message}`; }
}
refresh(); setInterval(refresh,1000);
</script></body></html>"""


app = create_app()
