"""The single-page live display. No build step, no external assets."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{WAKE_TITLE}} · Border Collie Voice</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0f1115; color: #e7e9ee;
         font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 12px;
           padding: 16px 20px; background: #161922; border-bottom: 1px solid #262b38; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  #status { margin-left: auto; font-size: 13px; display: flex; align-items: center; gap: 7px; }
  #dot { width: 9px; height: 9px; border-radius: 50%; background: #e0b341; }
  #dot.up { background: #35c26a; } #dot.down { background: #e5484d; }
  main { max-width: 820px; margin: 0 auto; padding: 20px; }
  .hint { color: #8b94a7; font-size: 14px; margin: 8px 2px 18px; }
  .event-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px; margin: 0 0 16px; }
  .event-panel { min-height: 150px; border-radius: 14px; padding: 18px 20px;
                 background: #161922; border: 1px solid #343a49; }
  .event-panel.heard { background: #14243d; border-color: #3c78c4; }
  .event-panel.valid { background: #10271a; border-color: #2e9b57;
                       box-shadow: 0 0 0 1px #2e9b5733; }
  .event-panel.invalid { background: #21171a; border-color: #8d4854; }
  .step { color: #8b94a7; font-size: 12px; font-weight: 700; letter-spacing: .1em;
          text-transform: uppercase; }
  .event-value { margin-top: 18px; color: #6b7385; font-size: 30px; font-weight: 800;
                 letter-spacing: .035em; }
  .event-panel.heard .event-value { color: #9bc2ff; }
  .event-panel.valid .event-value { color: #75e59d; }
  .event-panel.invalid .event-value { color: #e49aa5; }
  .event-detail { margin-top: 6px; color: #a9b2c3; font-size: 13px; }
  @media (max-width: 620px) { .event-grid { grid-template-columns: 1fr; } }
  .card { background: #161922; border: 1px solid #343a49;
          border-radius: 12px; padding: 18px 20px; margin: 0 0 12px; animation: pop .18s ease; }
  .card.valid { background: #10271a; border-color: #2e9b57; box-shadow: 0 0 0 1px #2e9b5733; }
  .card.invalid { background: #21171a; border-color: #684047; }
  @keyframes pop { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  .text { font-size: 28px; font-weight: 750; letter-spacing: .04em; }
  .card.valid .text { color: #75e59d; }
  .card.invalid .text { color: #d58d97; }
  .validity { margin-top: 5px; color: #91a09a; font-size: 12px; font-weight: 650;
              letter-spacing: .08em; text-transform: uppercase; }
  #empty { color: #6b7385; text-align: center; padding: 60px 0; }
  .history-title { margin: 24px 2px 10px; color: #8b94a7; font-size: 12px;
                   font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
  #actionControls { display: none; align-items: center; gap: 10px; margin: 0 0 16px;
                    padding: 12px 14px; border: 1px solid #4a3c1e; border-radius: 12px;
                    background: #211d13; color: #ead9a0; }
  #actionControls.show { display: flex; }
  #actionControls button { margin-left: auto; border: 0; border-radius: 9px; padding: 9px 13px;
                           background: #3577e5; color: white; font-weight: 650; cursor: pointer; }
  #actionControls button.armed { background: #b33f46; }
  #actionState { font-weight: 700; }
</style>
</head>
<body>
<header>
  <h1>{{WAKE_TITLE}} · Border Collie Voice</h1>
  <div id="status"><span id="dot"></span><span id="statusText">connecting...</span></div>
</header>
<main>
  <div class="hint">Say <b>&ldquo;{{WAKE}}&rdquo;</b>, then a command. Green means the command is valid for the dog; it does not mean the action executed.</div>
  <div class="event-grid">
    <section id="wakePanel" class="event-panel">
      <div class="step">1 &middot; Wake phrase</div>
      <div id="wakeValue" class="event-value">WAITING</div>
      <div id="wakeDetail" class="event-detail">Listening for &ldquo;{{WAKE}}&rdquo;</div>
    </section>
    <section id="commandPanel" class="event-panel">
      <div class="step">2 &middot; Command</div>
      <div id="commandValue" class="event-value">WAITING</div>
      <div id="commandDetail" class="event-detail">Say apple, banana, or pear after the wake phrase</div>
    </section>
  </div>
  <div id="actionControls">
    <span>Border Collie voice actions: <span id="actionState">disarmed</span>.<br>
      <small id="dogState">Checking the dog mission API…</small>
    </span>
    <button id="armButton" type="button">Arm voice actions</button>
  </div>
  <div class="history-title">Command history</div>
  <div id="feed"><div id="empty">Waiting for &ldquo;{{WAKE}}&rdquo;...</div></div>
</main>
<script>
  const feed = document.getElementById('feed');
  const dot = document.getElementById('dot');
  const statusText = document.getElementById('statusText');
  const wakePanel = document.getElementById('wakePanel');
  const wakeValue = document.getElementById('wakeValue');
  const wakeDetail = document.getElementById('wakeDetail');
  const commandPanel = document.getElementById('commandPanel');
  const commandValue = document.getElementById('commandValue');
  const commandDetail = document.getElementById('commandDetail');
  const actionControls = document.getElementById('actionControls');
  const actionState = document.getElementById('actionState');
  const dogState = document.getElementById('dogState');
  const armButton = document.getElementById('armButton');
  let empty = document.getElementById('empty');
  let currentCommandId = null;
  const WAKE_RESET_MS = 5000;
  let wakeResetTimer = null;

  function atTime(seconds) {
    return seconds ? new Date(seconds * 1000).toLocaleTimeString() : 'just now';
  }

  function resetCommand() {
    currentCommandId = null;
    commandPanel.className = 'event-panel';
    commandValue.textContent = 'WAITING';
    commandDetail.textContent = 'Wake phrase heard — listening for apple, banana, or pear';
  }

  function resetWake() {
    window.clearTimeout(wakeResetTimer);
    wakeResetTimer = null;
    wakePanel.className = 'event-panel';
    wakeValue.textContent = 'WAITING';
    wakeDetail.textContent = 'Listening for “{{WAKE}}”';
  }

  function renderWake(msg) {
    window.clearTimeout(wakeResetTimer);
    wakePanel.className = 'event-panel heard';
    wakeValue.textContent = (msg.wake_phrase || '{{WAKE}}').toUpperCase();
    wakeDetail.textContent = `HEARD at ${atTime(msg.detected_at)}`;
    resetCommand();
    wakeResetTimer = window.setTimeout(resetWake, WAKE_RESET_MS);
  }

  function renderCommand(msg) {
    const valid = msg.valid_dog_command === true;
    currentCommandId = msg.id || null;
    commandPanel.className = 'event-panel ' + (valid ? 'valid' : 'invalid');
    commandValue.textContent = valid
      ? (msg.display_command || '').toUpperCase()
      : 'NO VALID COMMAND';
    commandDetail.textContent = valid
      ? `VALID DOG COMMAND at ${atTime(msg.detected_at)}`
      : `Not a valid dog command at ${atTime(msg.detected_at)}`;
  }

  function renderAction(msg) {
    if (msg.id && currentCommandId && msg.id !== currentCommandId) return;
    if (msg.error) {
      commandDetail.textContent = `BLOCKED — ${msg.error}`;
      return;
    }
    if (msg.calls && msg.calls.length) {
      const result = msg.calls[0].result || 'accepted';
      commandDetail.textContent = `ACCEPTED BY DOG — ${result}`;
    }
  }

  function renderSnapshot(msg) {
    const voice = msg.voice || {};
    if (voice.wake) renderWake(voice.wake);
    if (voice.command) renderCommand(voice.command);
    if (voice.action) renderAction(voice.action);
  }

  function setStatus(up) {
    dot.className = up ? 'up' : 'down';
    statusText.textContent = up ? 'connected' : 'disconnected';
  }

  function addCard(msg) {
    if (empty) { empty.remove(); empty = null; }
    const isDogCommand = typeof msg.valid_dog_command === 'boolean';
    const valid = isDogCommand && msg.valid_dog_command;
    const card = document.createElement('div');
    card.className = 'card ' + (valid ? 'valid' : 'invalid');
    const text = document.createElement('div');
    text.className = 'text';
    text.textContent = isDogCommand
      ? (valid ? msg.display_command.toUpperCase() : 'NO VALID COMMAND')
      : msg.text;
    const validity = document.createElement('div');
    validity.className = 'validity';
    validity.textContent = valid ? 'Valid dog command' : 'Not a valid dog command';
    card.appendChild(text); card.appendChild(validity);
    feed.insertBefore(card, feed.firstChild);
    while (feed.children.length > 100) feed.removeChild(feed.lastChild);
  }

  function connect() {
    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.onopen = () => setStatus(true);
    ws.onclose = () => { setStatus(false); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();
    ws.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch (_) { return; }
      if (msg.kind === 'snapshot') renderSnapshot(msg);
      else if (msg.kind === 'wake') renderWake(msg);
      else if (msg.kind === 'command') { renderCommand(msg); addCard(msg); }
      else if (msg.kind === 'action') renderAction(msg);
    };
  }
  function renderActionState(status) {
    if (status.mode !== 'border_collie') return;
    actionControls.classList.add('show');
    if (!status.microphone || !status.microphone.ready) {
      const error = status.microphone?.error || 'microphone status unavailable';
      const missingReceiver = error.includes('no microphone matched');
      actionState.textContent = missingReceiver ? 'MIC NOT CONNECTED' : 'MIC ERROR';
      actionState.style.color = '#ff6b6b';
      dogState.textContent = missingReceiver
        ? 'DJI receiver not detected on Woof. Plug it in; listening will start automatically.'
        : error;
      armButton.disabled = true;
      armButton.textContent = 'Microphone unavailable';
      armButton.className = '';
      armButton.dataset.armed = '0';
      return;
    }
    if (!status.dog || !status.dog.ready) {
      const dogError = status.dog?.detail || 'dog mission status unavailable';
      actionState.textContent = status.armed ? 'ARMED / DOG BLOCKED' : 'DOG NOT READY';
      actionState.style.color = '#ff6b6b';
      dogState.textContent = dogError;
      armButton.disabled = !status.armed;
      armButton.textContent = status.armed ? 'Disarm voice actions' : 'Dog unavailable';
      armButton.className = status.armed ? 'armed' : '';
      armButton.dataset.armed = status.armed ? '1' : '0';
      return;
    }
    actionState.style.color = '';
    dogState.textContent = `${status.dog.build_label} ready at ${status.dog.url}`;
    armButton.disabled = false;
    actionState.textContent = status.armed
      ? (status.auto_arm_actions ? 'ARMED · ALWAYS ACTIVE' : 'ARMED')
      : 'disarmed';
    armButton.textContent = status.armed ? 'Disarm voice actions' : 'Arm voice actions';
    armButton.className = status.armed ? 'armed' : '';
    armButton.dataset.armed = status.armed ? '1' : '0';
  }
  async function refreshActionState() {
    try {
      const response = await fetch('/api/actions/status');
      renderActionState(await response.json());
    } catch (_) {}
  }
  armButton.addEventListener('click', async () => {
    const operation = armButton.dataset.armed === '1' ? 'disarm' : 'arm';
    armButton.disabled = true;
    try {
      const response = await fetch('/api/actions/' + operation, { method: 'POST' });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Voice action failed');
      renderActionState(body);
    } catch (error) {
      actionState.textContent = `ACTION ERROR: ${error.message}`;
      actionState.style.color = '#ff6b6b';
    } finally {
      await refreshActionState();
    }
  });
  connect();
  refreshActionState();
  setInterval(refreshActionState, 2000);
</script>
</body>
</html>
"""
