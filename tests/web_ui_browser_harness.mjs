import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, rm} from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

const [htmlPath, contractPath] = process.argv.slice(2);
const html = await readFile(htmlPath);
const tuningContract = JSON.parse(await readFile(contractPath, 'utf8'));

const terminalRun = {
  run_id: 'run-browser-001',
  activation_id: 'browser-activation-existing',
  target_fruit: 'apple',
  outcome: 'SUCCESS',
  reason: 'COMPLETED',
  message: 'Demo Run completed.',
  stage_results: {},
};
const api = {
  status: {
    build_label: 'browser-test',
    runtime_mode: 'simulation',
    active_run_id: null,
    mission: {restart_required: false},
    activation: {ready: true, blockers: []},
    run_tuning: tuningContract,
    fruit_bearing_map: {
      valid: true,
      invalidation_reason: null,
      last_observation_reason: 'centered_observations_recorded',
      anchor: {yaw_rad: Math.PI / 6},
      fruits: {
        apple: {
          bearing_rad: Math.PI / 2,
          confidence: 0.81,
          sample_count: 4,
          age_s: 1.25,
        },
        banana: {
          bearing_rad: -Math.PI / 6,
          confidence: 0.64,
          sample_count: 2,
          age_s: 2.5,
        },
        pear: {
          bearing_rad: Math.PI / 6,
          confidence: 0.72,
          sample_count: 3,
          age_s: 0.4,
        },
      },
      authority: 'yaw_route_only',
      reference: 'odometry_epoch_and_current_heading',
      persistence: 'process_local',
    },
  },
  cohort: null,
  run: {...terminalRun},
  runError: null,
  cohortError: null,
  runPosts: [],
  cohortPosts: [],
};

const experiments = {
  experiments: [
    {
      experiment_id: 'bearing-routing-flag-ab',
      name: 'Bearing-routing flag A/B',
      feature_under_test: 'Mapped bearing routing OFF versus ON',
      target_scope: 'Banana (controlled target)',
      status: 'COMPLETED',
      planned_runs: 2,
      attempted_runs: 2,
      successful_runs: 2,
      result: 'Routing ON reduced search rotation.',
    },
    {
      experiment_id: 'bearing-routing-on-random-ten',
      name: 'Bearing-routing ON randomized cohort',
      feature_under_test: 'Routing enabled for a ten-run cohort',
      target_scope: 'Apple, Banana, and Pear (randomized)',
      status: 'NOT_RUN',
      planned_runs: 10,
      attempted_runs: 0,
      successful_runs: 0,
      result: 'Planned; no result artifact exists yet.',
    },
  ],
  cohorts: [{
    target_fruit: 'banana',
    bearing_routing: 'ON',
    search_yaw_rps: 0.4,
    runs: 2,
    lock_rate_percent: 100,
    success_rate_percent: 100,
  }],
  note: 'Only terminal runs are included.',
  non_terminal_runs_excluded: 1,
};

function json(response, status = 200) {
  return {status, body: Buffer.from(JSON.stringify(response)), type: 'application/json'};
}

async function requestBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : null;
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  let result;
  if (request.method === 'GET' && url.pathname === '/') {
    result = {status: 200, body: html, type: 'text/html'};
  } else if (request.method === 'GET' && url.pathname === '/api/status') {
    result = json({...api.status, cohort: api.cohort});
  } else if (request.method === 'GET' && url.pathname === '/api/cohorts/active') {
    result = json({cohort: api.cohort});
  } else if (request.method === 'GET' && url.pathname === '/api/experiments/search') {
    result = json(experiments);
  } else if (request.method === 'GET' && url.pathname.startsWith('/api/results/')) {
    result = json({run: api.run});
  } else if (request.method === 'GET' && url.pathname === '/api/camera/frame.jpg') {
    result = {status: 404, body: Buffer.alloc(0), type: 'image/jpeg'};
  } else if (request.method === 'POST' && url.pathname === '/api/run') {
    const body = await requestBody(request);
    api.runPosts.push(body);
    if (api.runError) {
      result = json({detail: api.runError}, 409);
    } else {
      api.run = {
        run_id: 'run-browser-001',
        activation_id: body.activation_id,
        target_fruit: body.target_fruit,
        current_phase: 'preflight',
        message: 'Preflight in progress.',
        stage_results: {},
      };
      api.status.active_run_id = api.run.run_id;
      result = json({run: api.run, idempotent_replay: false}, 201);
    }
  } else if (request.method === 'POST' && url.pathname === '/api/cohorts') {
    const body = await requestBody(request);
    api.cohortPosts.push(body);
    if (api.cohortError) {
      result = json({detail: api.cohortError}, 409);
    } else {
      api.cohort = {
        cohort_id: `cohort-browser-${api.cohortPosts.length}`,
        status: 'RUNNING',
        policy: body,
        runs: [],
        home_scope: {kind: 'cohort'},
      };
      result = json({cohort: api.cohort}, 201);
    }
  } else {
    result = json({detail: `unexpected ${request.method} ${url.pathname}`}, 404);
  }
  response.writeHead(result.status, {
    'Content-Type': result.type,
    'Cache-Control': 'no-store',
    'Content-Length': result.body.length,
  });
  response.end(result.body);
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const address = server.address();
const pageUrl = `http://127.0.0.1:${address.port}/`;
const profile = await mkdtemp(path.join(os.tmpdir(), 'collie-ui-chrome-'));
const chromePath = process.env.CHROME_PATH
  ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const debugPort = 19000 + Math.floor(Math.random() * 1000);
const chrome = spawn(chromePath, [
  '--headless=new',
  '--disable-gpu',
  '--no-first-run',
  '--no-default-browser-check',
  `--remote-debugging-port=${debugPort}`,
  `--user-data-dir=${profile}`,
  pageUrl,
], {stdio: ['ignore', 'ignore', 'pipe']});

async function retry(work, timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      return await work();
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
  }
  throw lastError ?? new Error('retry timed out');
}

let socket;
let commandId = 0;
const commands = new Map();
async function cdp(method, params = {}) {
  const id = ++commandId;
  const response = new Promise((resolve, reject) => commands.set(id, {resolve, reject}));
  socket.send(JSON.stringify({id, method, params}));
  return response;
}

async function evaluate(expression) {
  const result = await cdp('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.exception?.description ?? 'browser evaluation failed');
  }
  return result.result.value;
}

async function waitFor(expression, timeoutMs = 5000) {
  return retry(async () => {
    const value = await evaluate(expression);
    if (!value) throw new Error(`condition not met: ${expression}`);
    return value;
  }, timeoutMs);
}

try {
  const target = await retry(async () => {
    const targets = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then((r) => r.json());
    const page = targets.find((item) => item.type === 'page' && item.url === pageUrl);
    if (!page) throw new Error('page target not ready');
    return page;
  });
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, {once: true});
    socket.addEventListener('error', reject, {once: true});
  });
  socket.addEventListener('message', ({data}) => {
    const message = JSON.parse(data);
    if (!message.id) return;
    const pending = commands.get(message.id);
    if (!pending) return;
    commands.delete(message.id);
    if (message.error) pending.reject(new Error(message.error.message));
    else pending.resolve(message.result);
  });
  await cdp('Runtime.enable');
  await cdp('Page.enable');
  await waitFor("document.querySelectorAll('#run-tuning-fields input').length > 20");

  // The production page must fit both a phone viewport and a desktop viewport.
  await cdp('Emulation.setDeviceMetricsOverride', {
    width: 320, height: 568, deviceScaleFactor: 1, mobile: true,
  });
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true);
  assert.equal(
    await evaluate(`(() => {
      const input = document.querySelector('#tuning-search-yaw_rps');
      const label = document.querySelector('label[for="tuning-search-yaw_rps"]');
      return input.getBoundingClientRect().top > label.getBoundingClientRect().top;
    })()`),
    true,
  );
  await cdp('Emulation.setDeviceMetricsOverride', {
    width: 1280, height: 800, deviceScaleFactor: 1, mobile: false,
  });
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true);
  assert.equal(
    await evaluate(`(() => {
      const input = document.querySelector('#tuning-search-yaw_rps');
      const label = document.querySelector('label[for="tuning-search-yaw_rps"]');
      const inputBox = input.getBoundingClientRect();
      const labelBox = label.getBoundingClientRect();
      return Math.abs(
        (inputBox.top + inputBox.height / 2) - (labelBox.top + labelBox.height / 2)
      ) < 2;
    })()`),
    true,
  );

  // Changing fruit must replace all target-bound defaults before editing.
  await evaluate(`(() => {
    const fruit = document.querySelector('#target-fruit');
    fruit.value = 'apple';
    fruit.dispatchEvent(new Event('change', {bubbles: true}));
  })()`);
  assert.deepEqual(
    await evaluate(`(() => {
      const names = ['focus_confidence', 'lock_confidence', 'tracking_confidence'];
      return names.map((name) => {
        const input = document.querySelector('#tuning-recognition-' + name);
        return [name, Number(input.value), Number(input.min), Number(input.max)];
      });
    })()`),
    [
      ['focus_confidence', 0.4, 0.4, 0.7],
      ['lock_confidence', 0.4, 0.4, 0.7],
      ['tracking_confidence', 0.1, 0.1, 0.7],
    ],
  );

  // Edit every generated setting and capture the independent expected payload.
  const expectedTuning = await evaluate(`(() => {
    const expected = {target_fruit: 'apple'};
    for (const input of document.querySelectorAll('#run-tuning-fields input')) {
      const [, group, ...parts] = input.id.split('-');
      const name = parts.join('-').replaceAll('-', '_');
      expected[group] ||= {};
      if (input.type === 'checkbox') {
        input.checked = !input.checked;
        expected[group][name] = input.checked;
      } else {
        const current = Number(input.value);
        const minimum = Number(input.min);
        const maximum = Number(input.max);
        const next = current === minimum ? maximum : minimum;
        input.value = String(next);
        expected[group][name] = name.endsWith('frames') || name.endsWith('confirmations')
          ? Number.parseInt(input.value, 10)
          : Number(input.value);
      }
      input.dispatchEvent(new Event('input', {bubbles: true}));
    }
    Object.defineProperty(window.crypto, 'randomUUID', {value: undefined, configurable: true});
    return expected;
  })()`);
  await evaluate("document.querySelector('#activate').click()");
  await retry(async () => assert.equal(api.runPosts.length, 1));
  assert.equal(api.runPosts[0].target_fruit, 'apple');
  assert.deepEqual(api.runPosts[0].tuning, expectedTuning);
  assert.match(api.runPosts[0].activation_id, /^audience-ui-/);
  assert.equal(new Set(api.runPosts.map((item) => item.activation_id)).size, 1);
  await waitFor(`
    document.querySelector('#run-id').textContent.includes('run-browser-001') &&
    document.querySelector('#run-tuning-controls').disabled
  `);
  assert.equal(await evaluate("document.querySelector('#activate').disabled"), true);
  assert.equal(await evaluate("document.querySelector('#target-fruit').disabled"), true);
  assert.equal(await evaluate("document.querySelector('#run-tuning-controls').disabled"), true);
  assert.equal(await evaluate("document.querySelector('#start-cohort').disabled"), true);

  // Terminal state relinquishes ownership and makes both run editors available.
  api.status.active_run_id = null;
  api.run = {...terminalRun};
  await waitFor("!document.querySelector('#activate').disabled", 3000);
  assert.equal(await evaluate("document.querySelector('#target-fruit').disabled"), false);
  assert.equal(await evaluate("document.querySelector('#run-tuning-controls').disabled"), false);
  assert.equal(await evaluate("document.querySelector('#start-cohort').disabled"), false);

  // A failed activation must remain visible across normal status polling.
  api.runError = 'run settings were rejected by the server';
  await evaluate("document.querySelector('#activate').click()");
  await retry(async () => assert.equal(api.runPosts.length, 2));
  await new Promise((resolve) => setTimeout(resolve, 1250));
  assert.match(await evaluate("document.querySelector('#status').textContent"), /run settings were rejected/);
  assert.equal(await evaluate("document.querySelector('#activate').disabled"), false);
  api.runError = null;

  // Fixed-fruit cohort payload includes every operator-editable policy field.
  await evaluate(`(() => {
    const randomized = document.querySelector('#cohort-randomized');
    randomized.checked = false;
    randomized.dispatchEvent(new Event('change', {bubbles: true}));
    document.querySelector('#cohort-runs').value = '7';
    document.querySelector('#cohort-fixed-fruit').value = 'pear';
    document.querySelector('#cohort-seed').value = '4242';
    [...document.querySelectorAll('[data-tolerated-reason]')]
      .find((control) => control.value === 'ARRIVAL_FAILURE').checked = true;
    [...document.querySelectorAll('[data-tolerated-phase]')]
      .find((control) => control.value === 'sit_and_bark').checked = true;
    document.querySelector('#start-cohort').click();
  })()`);
  await retry(async () => assert.equal(api.cohortPosts.length, 1));
  assert.deepEqual(api.cohortPosts[0], {
    runs: 7,
    randomized: false,
    target_fruit: 'pear',
    seed: 4242,
    tolerated_failures: [
      {reason: 'ARRIVAL_FAILURE'},
      {failed_phase: 'sit_and_bark'},
    ],
  });
  await waitFor("document.querySelector('#cohort-output').textContent.includes('RUNNING')");
  assert.equal(await evaluate("document.querySelector('#start-cohort').disabled"), true);
  assert.equal(await evaluate("document.querySelector('#activate').disabled"), true);
  assert.equal(await evaluate("document.querySelector('#target-fruit').disabled"), true);
  assert.equal(await evaluate("document.querySelector('#run-tuning-controls').disabled"), true);

  // Terminal cohort state returns all editors without a reload.
  api.cohort = {
    ...api.cohort,
    status: 'COMPLETED',
    runs: [{target_fruit: 'pear', outcome: 'SUCCESS', reason: 'COMPLETED'}],
  };
  await waitFor("!document.querySelector('#start-cohort').disabled", 3000);
  assert.equal(await evaluate("document.querySelector('#activate').disabled"), false);

  // Randomized mode deliberately sends no fixed Target Fruit.
  await evaluate(`(() => {
    const randomized = document.querySelector('#cohort-randomized');
    randomized.checked = true;
    randomized.dispatchEvent(new Event('change', {bubbles: true}));
    document.querySelector('#cohort-runs').value = '10';
    document.querySelector('#cohort-seed').value = '20260814';
    [...document.querySelectorAll('[data-tolerated-reason]')]
      .find((control) => control.value === 'ARRIVAL_FAILURE').checked = false;
    [...document.querySelectorAll('[data-tolerated-phase]')]
      .find((control) => control.value === 'sit_and_bark').checked = false;
    document.querySelector('#start-cohort').click();
  })()`);
  await retry(async () => assert.equal(api.cohortPosts.length, 2));
  assert.deepEqual(api.cohortPosts[1], {
    runs: 10,
    randomized: true,
    target_fruit: null,
    seed: 20260814,
    tolerated_failures: [],
  });

  // Cohort API errors remain visible and release the pending click lock.
  api.cohort = {...api.cohort, status: 'COMPLETED'};
  await waitFor("!document.querySelector('#start-cohort').disabled", 3000);
  api.cohortError = 'cohort policy conflicts with active ownership';
  await evaluate("document.querySelector('#start-cohort').click()");
  await retry(async () => assert.equal(api.cohortPosts.length, 3));
  await new Promise((resolve) => setTimeout(resolve, 1250));
  assert.match(await evaluate("document.querySelector('#cohort-output').textContent"), /cohort policy conflicts/);
  assert.equal(await evaluate("document.querySelector('#start-cohort').disabled"), false);

  // Home semantics and evidence rows must be explicit and accurately distinct.
  assert.match(
    await evaluate("document.querySelector('[aria-labelledby=cohort-title] > p').textContent"),
    /captures a new Home for this cohort.*same stored Home/,
  );
  const evidenceText = await evaluate("document.querySelector('#experiment-evidence').textContent");
  assert.match(evidenceText, /Bearing-routing flag A\/B/);
  assert.match(evidenceText, /Banana \(controlled target\)/);
  assert.match(evidenceText, /Not run/);
  assert.match(await evaluate("document.querySelector('#experiment-live-note').textContent"), /1 non-terminal run/);

  // Bearing visualization is a read-only view of the existing process-local map.
  await waitFor("document.querySelector('#fruit-bearing-list [data-bearing-fruit=apple]')?.textContent.includes('+60.0° left')");
  assert.match(
    await evaluate("document.querySelector('#fruit-bearing-list [data-bearing-fruit=apple]').textContent"),
    /81\.0%.*4 samples.*1\.3 s/,
  );
  assert.match(
    await evaluate("document.querySelector('#fruit-bearing-list [data-bearing-fruit=banana]').textContent"),
    /-60\.0° right.*64\.0%.*2 samples.*2\.5 s/,
  );
  assert.match(
    await evaluate("document.querySelector('#fruit-bearing-list [data-bearing-fruit=pear]').textContent"),
    /0\.0° aligned.*72\.0%.*3 samples.*0\.4 s/,
  );
  assert.equal(
    await evaluate("document.querySelector('#fruit-bearing-map').dataset.authority"),
    'yaw_route_only',
  );
  assert.match(
    await evaluate("document.querySelector('#fruit-bearing-note').textContent"),
    /visualization only.*never authorizes motion/i,
  );

  console.log('browser UI contract: PASS');
} finally {
  if (socket) socket.close();
  chrome.kill('SIGTERM');
  server.close();
  await new Promise((resolve) => {
    if (chrome.exitCode !== null) resolve();
    else chrome.once('exit', resolve);
  });
  await retry(() => rm(profile, {recursive: true, force: true}));
}
