// Electron main process for Sunday — the "app shell".
// On launch it makes sure the Python backend (FastAPI) is running, shows the glowing orb,
// and puts an icon in the system tray. The tray menu lets you turn hands-free on/off, open
// the dashboard, or quit. Clicking the orb opens the dashboard too.
//
// It only starts the backend if one isn't already running, so it won't clash with a uvicorn
// you started yourself.

const { app, BrowserWindow, Tray, Menu, ipcMain, screen } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const REPO = path.join(__dirname, '..');
const API = 'http://localhost:8000';

let sidecar = null;
let orbWin = null;
let dashWin = null;
let tray = null;

// ---- tiny HTTP helpers (talk to the FastAPI backend) ----
function ping(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => { res.resume(); resolve(res.statusCode === 200); });
    req.on('error', () => resolve(false));
    req.setTimeout(800, () => { req.destroy(); resolve(false); });
  });
}
function post(pathname) {
  return new Promise((resolve) => {
    const req = http.request(API + pathname, { method: 'POST' }, (res) => { res.resume(); resolve(res.statusCode); });
    req.on('error', () => resolve(0));
    req.end();
  });
}
function getJSON(pathname) {
  return new Promise((resolve) => {
    const req = http.get(API + pathname, (res) => {
      let buf = ''; res.on('data', (c) => (buf += c));
      res.on('end', () => { try { resolve(JSON.parse(buf)); } catch { resolve(null); } });
    });
    req.on('error', () => resolve(null));
    req.setTimeout(800, () => { req.destroy(); resolve(null); });
  });
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- backend lifecycle ----
async function ensureBackend() {
  if (await ping(API + '/health')) { console.log('[sunday] backend already running'); return; }
  const py = process.platform === 'win32'
    ? path.join(REPO, 'venv', 'Scripts', 'python.exe')
    : path.join(REPO, 'venv', 'bin', 'python');
  console.log('[sunday] starting backend:', py);
  sidecar = spawn(py, ['-m', 'uvicorn', 'app.main:app', '--port', '8000'],
                  { cwd: REPO, stdio: 'inherit' });
  sidecar.on('error', (e) => console.error('[sunday] failed to start backend:', e.message));
  for (let i = 0; i < 60; i++) {
    if (await ping(API + '/health')) { console.log('[sunday] backend up'); return; }
    await sleep(500);
  }
  console.error('[sunday] backend did not respond in time');
}

// ---- windows ----
function createOrb() {
  orbWin = new BrowserWindow({
    width: 300, height: 300,
    frame: false, transparent: true, resizable: false,
    alwaysOnTop: true, skipTaskbar: true, hasShadow: false,
    webPreferences: { preload: path.join(__dirname, 'preload.js'),
                      contextIsolation: true, nodeIntegration: false },
  });
  orbWin.loadFile(path.join(__dirname, 'orb.html'));
  // Pin to the TOP-right corner of the work area (orb itself is top/right-aligned).
  const { workArea } = screen.getPrimaryDisplay();
  orbWin.setPosition(workArea.x + workArea.width - 300, workArea.y);
  orbWin.on('closed', () => { orbWin = null; });
}

function openDashboard() {
  if (dashWin && !dashWin.isDestroyed()) { dashWin.show(); dashWin.focus(); return; }
  dashWin = new BrowserWindow({
    width: 440, height: 580, title: 'Sunday',
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  dashWin.loadFile(path.join(__dirname, 'dashboard.html'));
  dashWin.on('closed', () => { dashWin = null; });
}

function toggleOrb() {
  if (!orbWin) return createOrb();
  orbWin.isVisible() ? orbWin.hide() : orbWin.show();
}

// ---- tray ----
function buildTray() {
  tray = new Tray(path.join(__dirname, 'assets', 'tray.png'));
  tray.setToolTip('Sunday');
  tray.on('click', openDashboard);                         // left-click → dashboard
  tray.on('right-click', () => tray.popUpContextMenu());   // explicit (Windows robustness)
  updateTray({ running: false, state: 'idle' });
}

function updateTray(d) {
  const running = !!d.running;
  const orbShown = orbWin && orbWin.isVisible();
  const menu = Menu.buildFromTemplate([
    { label: `Sunday — ${running ? '● ' + (d.state || 'on') : 'off'}`, enabled: false },
    { type: 'separator' },
    running
      ? { label: 'Stop listening', click: () => post('/wake/stop') }
      : { label: 'Start listening  (say "Sunday")', click: () => post('/wake/start') },
    { label: 'Open dashboard', click: openDashboard },
    { label: orbShown ? 'Hide orb' : 'Show orb', click: toggleOrb },
    { type: 'separator' },
    { label: 'Quit Sunday', click: () => { app.isQuitting = true; app.quit(); } },
  ]);
  tray.setContextMenu(menu);
  tray.setToolTip(`Sunday — ${running ? (d.state || 'on') : 'off'}`);
}

// ---- IPC from the orb renderer ----
ipcMain.on('open-dashboard', openDashboard);
ipcMain.on('set-wake', (_e, on) => post(on ? '/wake/start' : '/wake/stop'));
ipcMain.on('quit-app', () => { app.isQuitting = true; app.quit(); });

// ---- app lifecycle ----
app.whenReady().then(async () => {
  await ensureBackend();
  createOrb();
  buildTray();
  // Keep the tray menu + tooltip in sync with the backend state.
  setInterval(async () => { const d = await getJSON('/state'); if (d) updateTray(d); }, 1000);
});

// Stay alive in the tray when windows are closed (quit only from the tray menu).
app.on('window-all-closed', (e) => { /* don't quit */ });
app.on('before-quit', () => { if (sidecar) sidecar.kill(); });
