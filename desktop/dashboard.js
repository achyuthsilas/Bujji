// Dashboard logic: pull todos / reminders / notes from the backend and render each section.
const API = 'http://localhost:8000';

function render(listId, countId, items, fmt) {
  const ul = document.getElementById(listId);
  document.getElementById(countId).textContent = items.length ? `(${items.length})` : '';
  ul.innerHTML = '';
  if (!items.length) {
    ul.innerHTML = '<li class="empty">Nothing yet — add some by voice.</li>';
    return;
  }
  for (const it of items) ul.appendChild(fmt(it));
}

function row(text, when) {
  const li = document.createElement('li');
  const t = document.createElement('span'); t.textContent = text;
  li.appendChild(t);
  if (when) { const w = document.createElement('span'); w.className = 'when'; w.textContent = when; li.appendChild(w); }
  return li;
}

async function load() {
  const status = document.getElementById('status');
  try {
    const [todos, reminders, notes] = await Promise.all([
      fetch(API + '/todos').then((r) => r.json()),
      fetch(API + '/reminders').then((r) => r.json()),
      fetch(API + '/notes').then((r) => r.json()),
    ]);
    render('todos', 'todos-count', todos.items, (t) => row(t.task, t.done ? 'done' : ''));
    render('reminders', 'reminders-count', reminders.items, (r) => row(r.content, r.remind_at || ''));
    render('notes', 'notes-count', notes.items, (n) => row(n.content, ''));
    status.textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    status.textContent = 'Could not reach Sunday backend (' + e.message + ')';
  }
}

document.getElementById('refresh').addEventListener('click', load);
load();
