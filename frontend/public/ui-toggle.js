/* Old/new UI switch, shared by every Optivail app.
 * Include in <head> of both UIs:
 *   <script src="/ui-toggle.js" data-old="/" data-new="/v2/index.html"></script>
 * Preference lives in localStorage; ?ui=old|new overrides and is remembered.
 * Default is the old UI. */
(function () {
  var KEY = 'optivail-ui';
  var me = document.currentScript;
  var OLD = (me && me.dataset.old) || '/';
  var NEW = (me && me.dataset.new) || '/v2/index.html';
  var onNew = location.pathname.indexOf(NEW.replace(/index\.html$/, '')) === 0 && NEW !== OLD;

  function read() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function write(v) { try { localStorage.setItem(KEY, v); } catch (e) { /* private mode */ } }

  var params = new URLSearchParams(location.search);
  var forced = params.get('ui');
  if (forced === 'old' || forced === 'new') write(forced);
  var pref = read() === 'new' ? 'new' : 'old';

  function go(target) {
    var url = new URL(target, location.origin);
    params.delete('ui');
    var q = params.toString();
    location.replace(url.pathname + (q ? '?' + q : '') + location.hash);
  }

  if (pref === 'new' && !onNew) { go(NEW); return; }
  if (pref === 'old' && onNew) { go(OLD); return; }

  function mount() {
    if (document.getElementById('ui-toggle')) return;
    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var wrap = document.createElement('div');
    wrap.id = 'ui-toggle';
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', 'Interface version');
    wrap.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:2147483000;display:flex;gap:2px;padding:3px;' +
      'border-radius:999px;background:rgba(13,17,23,.82);border:1px solid rgba(255,255,255,.12);' +
      'backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);box-shadow:0 6px 20px rgba(0,0,0,.35);' +
      'font:600 11px/1 Inter,system-ui,sans-serif;letter-spacing:.02em';
    [['old', 'Oud'], ['new', 'Nieuw']].forEach(function (o) {
      var b = document.createElement('button');
      var active = (o[0] === 'new') === onNew;
      b.type = 'button';
      b.textContent = o[1];
      b.setAttribute('aria-pressed', active ? 'true' : 'false');
      b.style.cssText = 'border:0;cursor:pointer;padding:6px 11px;border-radius:999px;font:inherit;' +
        (reduce ? '' : 'transition:background .2s,color .2s;') +
        (active ? 'background:#e6edf3;color:#0d1117;' : 'background:transparent;color:#9aa4af;');
      b.addEventListener('click', function () {
        if (active) return;
        write(o[0]);
        go(o[0] === 'new' ? NEW : OLD);
      });
      wrap.appendChild(b);
    });
    document.body.appendChild(wrap);
  }
  if (document.body) mount(); else document.addEventListener('DOMContentLoaded', mount);
})();
