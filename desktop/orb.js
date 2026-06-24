// Orb logic: poll the backend's /state ~4x/sec, drive the glow, and pop out the
// transcript + reply whenever a new turn completes. Click the orb to open the dashboard.

const API = 'http://localhost:8000';
const body = document.body;
const label = document.getElementById('label');
const bubble = document.getElementById('bubble');
const bYou = document.getElementById('b-you');
const bSun = document.getElementById('b-sun');
const bMeta = document.getElementById('b-meta');

let lastTurn = 0;
let hideTimer = null;
let running = false;     // is hands-free listening on?

// The orb is a drag handle (move the window). Open the dashboard from the hover menu.

// Hover menu buttons.
const mWake = document.getElementById('m-wake');
mWake.addEventListener('click', () => window.sunday?.setWake(!running));
document.getElementById('m-dash').addEventListener('click', () => window.sunday?.openDashboard());
document.getElementById('m-quit').addEventListener('click', () => window.sunday?.quit());

function showBubble(d) {
  bYou.textContent = d.transcript || '(didn’t catch that)';
  bSun.textContent = d.reply || '…';
  const t = (d.wake_to_answer_ms != null) ? `${d.wake_to_answer_ms} ms`
          : (d.first_audio_ms != null) ? `${d.first_audio_ms} ms` : '';
  const steps = ['stt_ms', 'llm_ms', 'tts_ms']
    .filter((k) => d[k] != null).map((k) => `${k.replace('_ms', '').toUpperCase()} ${d[k]}`)
    .join(' · ');
  bMeta.textContent = [t && `answered in ${t}`, steps].filter(Boolean).join('  ·  ');
  bubble.classList.remove('hidden');
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => bubble.classList.add('hidden'), 7000);
}

async function poll() {
  try {
    const res = await fetch(API + '/state');
    const d = await res.json();

    // Glow state on <body>.
    const state = d.state || 'idle';
    body.className = state;
    running = !!d.running;
    label.textContent = running ? state : 'off';
    mWake.textContent = running ? '🛑 Stop listening' : '🎤 Start listening';

    // New completed turn? Pop the bubble.
    if (d.turn && d.turn !== lastTurn && d.transcript !== undefined) {
      lastTurn = d.turn;
      showBubble(d);
    }
  } catch (e) {
    body.className = 'idle';
    running = false;
    label.textContent = '…';   // backend not reachable yet
    mWake.textContent = '🎤 Start listening';
  }
}

setInterval(poll, 250);
poll();
