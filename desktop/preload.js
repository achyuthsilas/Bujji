// Safe bridge between the orb window (renderer) and the main process.
// Exposes a tiny `window.sunday` API so the orb can open the dashboard / toggle hands-free
// without giving the page full Node access.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('sunday', {
  openDashboard: () => ipcRenderer.send('open-dashboard'),
  setWake: (on) => ipcRenderer.send('set-wake', on),
  quit: () => ipcRenderer.send('quit-app'),
});
