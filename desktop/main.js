// Electron main process for Sunday — the "app shell".
// Responsibilities:
//   • make sure the Python backend (FastAPI) is running, and reliably clean it up
//   • show the glowing orb + system-tray icon
//   • remember the orb's position and (optionally) start on Windows login
//
// Backend lifecycle (so a backend never lingers):
//   - On launch we kill any orphaned backend we started in a previous run (tracked by a
//     PID file), then start a fresh one — UNLESS a backend you started yourself is already
//     on :8000, in which case we just reuse it and leave it alone.
//   - On quit we kill the backend we started.

const { app, BrowserWindow, Tray, Menu, ipcMain, screen } = require('electron');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { spawn } = require('child_process');

const REPO = path.join(__dirname, '..');
const API = 'http://localhost:8000';
const isDev = !app.isPackaged;

let sidecar = null;      // backend process WE started (null if we reused an external one)
let orbWin = null;
let dashWin = null;
let tray = null;
let SETTINGS_FILE = null;
let PID_FILE = null;
let settings = {};       // { orb: {x,y}, openAtLogin: bool }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- settings persistence (small JSON in the app's userData folder) ----
function loadSettings() { try { settings = JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf8')); } catch { settings = {}; } }
function saveSettings() { try { fs.writeFileSync(SETTINGS_FILE, JSON.stringify(settings, null, 2)); } catch {} }

// ---- HTTP helpers (talk to the backend) ----
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

// ---- backend process management ----
function isAlive(pid) { try { process.kill(pid, 0); return true; } catch { return false; } }
function killPid(pid) {
  if (!pid) return;
  try { process.kill(pid); } catch {}
  if (process.platform === 'win32') { try { spawn('taskkill', ['/PID', String(pid), '/T', '/F']); } catch {} }
}
function readPid() { try { return parseInt(fs.readFileSync(PID_FILE, 'utf8'), 10); } catch { return null; } }

async function ensureBackend() {
  if (await ping(API + '/health')) { console.log('[sunday] reusing a backend already on :8000'); return; }
  const py = process.platform === 'win32'
    ? path.join(REPO, 'venv', 'Scripts', 'python.exe')
    : path.join(REPO, 'venv', 'bin', 'python');
  console.log('[sunday] starting backend:', py);
  sidecar = spawn(py, ['-m', 'uvicorn', 'app.main:app', '--port', '8000'], { cwd: REPO, stdio: 'inherit' });
  sidecar.on('error', (e) => console.error('[sunday] failed to start backend:', e.message));
  try { fs.writeFileSync(PID_FILE, String(sidecar.pid)); } catch {}
  for (let i = 0; i < 60; i++) {
    if (await ping(API + '/health')) { console.log('[sunday] backend up'); return; }
    await sleep(500);
  }
  console.error('[sunday] backend did not respond in time');
}

// ---- auto-start on login ----
function getOpenAtLogin() { return app.getLoginItemSettings().openAtLogin; }
function setOpenAtLogin(on) {
  const opts = { openAtLogin: on };
  if (isDev) { opts.path = process.execPath; opts.args = [path.resolve(__dirname)]; }  // launch this app folder
  app.setLoginItemSettings(opts);
  settings.openAtLogin = on; saveSettings();
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

  // Restore saved position, else top-right. Clamp into the work area so it can't get lost.
  const wa = screen.getPrimaryDisplay().workArea;
  let x = (settings.orb && Number.isInteger(settings.orb.x)) ? settings.orb.x : (wa.x + wa.width - 300);
  let y = (settings.orb && Number.isInteger(settings.orb.y)) ? settings.orb.y : wa.y;
  x = Math.min(Math.max(x, wa.x), wa.x + wa.width - 60);
  y = Math.min(Math.max(y, wa.y), wa.y + wa.height - 60);
  orbWin.setPosition(Math.round(x), Math.round(y));

  // Remember where you drag it to.
  orbWin.on('moved', () => { const [px, py] = orbWin.getPosition(); settings.orb = { x: px, y: py }; saveSettings(); });
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
  tray.on('click', openDashboard);
  tray.on('right-click', () => tray.popUpContextMenu());
  updateTray({ running: false, state: 'idle' });
}
function updateTray(d) {
  const running = !!d.running;
  const orbShown = orbWin && orbWin.isVisible();
  const menu = Menu.buildFromTemplate([
    { label: `Sunday — ${running ? '● ' + (d.state || 'on') : 'off'}`, enabled: false },
    { type: 'separator' },
    running ? { label: 'Stop listening', click: () => post('/wake/stop') }
            : { label: 'Start listening  (say "Sunday")', click: () => post('/wake/start') },
    { label: 'Open dashboard', click: openDashboard },
    { label: orbShown ? 'Hide orb' : 'Show orb', click: toggleOrb },
    { type: 'checkbox', label: 'Start at login', checked: getOpenAtLogin(), click: (mi) => setOpenAtLogin(mi.checked) },
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
  SETTINGS_FILE = path.join(app.getPath('userData'), 'settings.json');
  PID_FILE = path.join(app.getPath('userData'), 'backend.pid');
  loadSettings();

  // Clean up a backend orphaned by a previous run (e.g. after a crash).
  const orphan = readPid();
  if (orphan && isAlive(orphan)) { console.log('[sunday] cleaning orphaned backend', orphan); killPid(orphan); await sleep(1200); }
  try { fs.unlinkSync(PID_FILE); } catch {}

  // First run: default to starting on login (you asked for "always there").
  if (settings.openAtLogin === undefined) settings.openAtLogin = true;
  setOpenAtLogin(settings.openAtLogin);

  await ensureBackend();
  createOrb();
  buildTray();
  setInterval(async () => { const d = await getJSON('/state'); if (d) updateTray(d); }, 1000);
});

// Stay alive in the tray when windows are closed (quit only from the menu).
app.on('window-all-closed', () => {});
app.on('before-quit', () => { if (sidecar) killPid(sidecar.pid); try { fs.unlinkSync(PID_FILE); } catch {} });
