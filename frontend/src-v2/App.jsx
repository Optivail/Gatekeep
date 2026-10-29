import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import WikiPage from '../src/Wiki.jsx'

const POLL = 3000
const PAGE_SIZE = 40
const RANGES = [
  { id: '1h', label: '1h' },
  { id: '24h', label: '24h' },
  { id: '7d', label: '7d' },
  { id: '30d', label: '30d' },
  { id: 'all', label: 'All' },
]
const TOGGLES = [
  { key: 'compression_enabled', label: 'Compression', hint: 'Payload optimize + response cache' },
  { key: 'anti_yap_enabled', label: 'Anti-yap', hint: 'Shorter output estimates' },
  { key: 'history_pruning_enabled', label: 'History pruning', hint: 'Trim conversation history' },
  { key: 'log_truncation_enabled', label: 'Log truncation', hint: 'Errors + head/tail only' },
  { key: 'aggressive_enabled', label: 'Aggressive', hint: 'Maximum savings mode; forces pruning and truncation on' },
]
const AGENT_COLORS = { claude: '#f59e0b', grok: '#38bdf8', openai: '#10b981', cursor: '#a855f7', gemini: '#3b82f6', mistral: '#fb923c', devin: '#06b6d4', freebuff: '#ec4899', other: '#71717a' }
const agentColor = (id) => AGENT_COLORS[id] || AGENT_COLORS.other

const fmt = (n) => {
  const v = Number(n ?? 0)
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 10_000) return `${(v / 1000).toFixed(1)}k`
  return Math.round(v).toLocaleString()
}
const fmtUsd = (n) => `$${Number(n ?? 0).toFixed(2)}`
const fmtRate = (n) => { const v = Number(n ?? 0); return v >= 1 ? `$${v.toFixed(2)}` : v >= 0.01 ? `$${v.toFixed(3)}` : `$${v.toFixed(4)}` }
function parseTs(raw) {
  if (!raw) return null
  const s = String(raw).trim().replace(' ', 'T')
  const d = new Date(s.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(s) ? s : `${s}Z`)
  return Number.isNaN(d.getTime()) ? null : d
}
const fmtTime = (iso) => parseTs(iso)?.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) ?? '—'
const fmtLogTime = (iso) => parseTs(iso)?.toLocaleString(undefined, { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) ?? '—'
function ago(iso) {
  const d = parseTs(iso); if (!d) return ''
  const s = Math.max(0, Math.round((Date.now() - d) / 1000))
  return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.round(s / 60)}m ago` : s < 86400 ? `${Math.round(s / 3600)}h ago` : `${Math.round(s / 86400)}d ago`
}
const savePct = (l) => (l?.original_input_tokens ? (l.input_tokens_saved / l.original_input_tokens) * 100 : 0)
const modelLabel = (r) => (r.model || r.tag || r.provider || '—').slice(0, 40)
function resetIn(epochSec) {
  if (!epochSec) return '—'
  const ms = epochSec * 1000 - Date.now()
  if (ms <= 0) return 'resetting soon'
  const m = Math.round(ms / 60000)
  return m < 60 ? `in ${m}m` : `in ${Math.floor(m / 60)}h ${m % 60}m`
}

const METRICS = [
  { id: 'sent', label: 'Tokens used', card: (s) => fmt(s?.total_optimized_input_tokens), sub: (s) => { const r = s?.total_original_input_tokens || 0; return r ? `${((s.total_optimized_input_tokens / r) * 100).toFixed(0)}% of raw forwarded` : 'after compression' }, bucket: (b) => b.optimized_input_tokens || 0, fmt, chart: 'area' },
  { id: 'saved', label: 'Tokens saved', card: (s) => fmt(s?.total_tokens_saved), delta: (c) => c?.tokens_saved_pct, bucket: (b) => b.input_tokens_saved || 0, fmt, chart: 'bar' },
  { id: 'rate', label: 'Save rate', card: (s) => `${Number(s?.avg_save_pct ?? 0).toFixed(1)}%`, delta: (c) => c?.save_rate_pct, bucket: (b) => { const o = b.original_input_tokens || 0; return o ? ((b.input_tokens_saved || 0) / o) * 100 : 0 }, fmt: (v) => `${v.toFixed(1)}%`, chart: 'area' },
  { id: 'prompt', label: 'Prompt tokens', card: (s) => fmt(s?.total_original_input_tokens), delta: (c) => c?.original_input_pct, bucket: (b) => b.original_input_tokens || 0, fmt, chart: 'area' },
  { id: 'requests', label: 'Requests', card: (s) => fmt(s?.total_requests), delta: (c) => c?.requests_pct, bucket: (b) => b.requests || 0, fmt, chart: 'bar' },
  { id: 'usd', label: 'Est. saved', card: (s) => fmtUsd(s?.estimated_usd_saved), delta: (c) => c?.usd_saved_pct, bucket: (b) => b.input_tokens_saved || 0, fmt: (v) => `${fmt(v)} tok`, chart: 'bar' },
]

/* ── Motion primitives ─────────────────────────────────────────────── */
const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
function useTween(target, dur = 700) {
  const [v, setV] = useState(target)
  const from = useRef(target)
  useEffect(() => {
    const start = performance.now(), a = from.current, b = Number(target) || 0
    if (reduceMotion() || a === b) { from.current = b; setV(b); return }
    let raf
    const step = (t) => {
      const k = Math.min((t - start) / dur, 1), e = 1 - Math.pow(1 - k, 4)
      const cur = a + (b - a) * e
      from.current = cur; setV(cur)
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, dur])
  return v
}
function Num({ value, format = fmt, dur }) {
  const v = useTween(Number(value) || 0, dur)
  return <>{format(v)}</>
}

/* Segmented control with a sliding indicator. */
function Seg({ items, value, onChange, label, size }) {
  const wrap = useRef(null)
  const [ind, setInd] = useState(null)
  useLayoutEffect(() => {
    const el = wrap.current?.querySelector('[aria-pressed="true"]')
    if (el) setInd({ left: el.offsetLeft, width: el.offsetWidth })
  }, [value, items.length])
  useEffect(() => {
    const on = () => { const el = wrap.current?.querySelector('[aria-pressed="true"]'); if (el) setInd({ left: el.offsetLeft, width: el.offsetWidth }) }
    window.addEventListener('resize', on); document.fonts?.ready.then(on)
    return () => window.removeEventListener('resize', on)
  }, [])
  return (
    <div className={`seg ${size || ''}`} role="group" aria-label={label} ref={wrap}>
      {ind && <span className="seg-ind" style={{ transform: `translateX(${ind.left}px)`, width: ind.width }} />}
      {items.map((it) => (
        <button key={it.id} type="button" aria-pressed={value === it.id} onClick={() => onChange(it.id)}>
          {it.dot && <i style={{ background: it.dot }} />}{it.label}
        </button>
      ))}
    </div>
  )
}

function Switch({ checked, disabled, onChange, label }) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} disabled={disabled} className={`switch ${checked ? 'on' : ''}`} onClick={() => !disabled && onChange(!checked)}>
      <span />
    </button>
  )
}

/* ── Charts ────────────────────────────────────────────────────────── */
function Spark({ values, color = 'currentColor' }) {
  const pts = values?.length > 1 ? values : [0, 0]
  const max = Math.max(...pts, 1), min = Math.min(...pts, 0), span = Math.max(max - min, 1), step = 100 / (pts.length - 1)
  const d = pts.map((v, i) => `${i ? 'L' : 'M'}${(i * step).toFixed(1)},${(30 - ((v - min) / span) * 26 - 2).toFixed(1)}`).join(' ')
  return <svg viewBox="0 0 100 32" className="spark" preserveAspectRatio="none" aria-hidden="true"><path d={d} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" pathLength="1" className="draw" /></svg>
}

function Chart({ series, metric }) {
  const ref = useRef(null)
  const [hi, setHi] = useState(null)
  useEffect(() => setHi(null), [series, metric])
  if (!series?.length) return <div className="empty"><b>No traffic in this period.</b><span>Point a CLI at Gatekeep (Settings → Docs) and requests show up here live.</span></div>
  const values = series.map(metric.bucket), max = Math.max(...values, 1), n = series.length
  const step = n > 1 ? 100 / (n - 1) : 100
  const pts = values.map((v, i) => [n > 1 ? i * step : 50, 100 - (v / max) * 88 - 4])
  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect()
    setHi(n <= 1 ? 0 : Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1)))))
  }
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ')
  const tip = hi != null ? series[hi] : null
  return (
    <div className="chart" ref={ref} onMouseMove={onMove} onMouseLeave={() => setHi(null)} key={metric.id}>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {[25, 50, 75].map((y) => <line key={y} x1="0" x2="100" y1={y} y2={y} className="grid" vectorEffect="non-scaling-stroke" />)}
        {metric.chart === 'bar'
          ? pts.map(([x, y], i) => { const w = Math.max(step * 0.62, 0.8); return <rect key={i} x={x - w / 2} y={y} width={w} height={100 - y} rx="0.6" className={`bar ${i === hi ? 'hi' : ''}`} style={{ '--i': i }} /> })
          : <>
              <defs><linearGradient id="ga" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--ink)" stopOpacity=".16" /><stop offset="1" stopColor="var(--ink)" stopOpacity="0" /></linearGradient></defs>
              <path d={`${line} L${pts[n - 1][0]},100 L${pts[0][0]},100 Z`} fill="url(#ga)" className="fade-in" />
              <path d={line} fill="none" className="line draw" pathLength="1" vectorEffect="non-scaling-stroke" />
            </>}
        {hi != null && <line x1={pts[hi][0]} x2={pts[hi][0]} y1="0" y2="100" className="cross" vectorEffect="non-scaling-stroke" />}
      </svg>
      {hi != null && <span className="chart-dot" style={{ left: `${pts[hi][0]}%`, top: `${pts[hi][1]}%` }} />}
      {tip && (
        <div className="chart-tip" style={{ left: `${Math.min(88, Math.max(12, pts[hi][0]))}%` }}>
          <span className="t">{fmtTime(tip.t)}</span>
          <b className="mono">{metric.fmt(values[hi])}</b>
          <span className="mono s">{fmt(tip.requests || 0)} req · raw {fmt(tip.original_input_tokens)} · sent {fmt(tip.optimized_input_tokens)}</span>
        </div>
      )}
    </div>
  )
}

function Ring({ pct, color, size = 76 }) {
  const p = useTween(Math.max(0, Math.min(100, pct || 0)), 900), r = 30, c = 2 * Math.PI * r
  return (
    <svg width={size} height={size} viewBox="0 0 76 76" className="ring" aria-hidden="true">
      <circle cx="38" cy="38" r={r} className="ring-bg" />
      <circle cx="38" cy="38" r={r} className="ring-fg" style={{ stroke: color, strokeDasharray: `${(c * p) / 100} ${c}` }} />
      <text x="38" y="42" textAnchor="middle" className="mono">{p.toFixed(0)}%</text>
    </svg>
  )
}
const loadColor = (p, base) => (p > 80 ? 'var(--bad)' : p > 50 ? 'var(--warn)' : base)

/* ── Sections ──────────────────────────────────────────────────────── */
function Hero({ stats }) {
  const raw = stats?.total_original_input_tokens || 0, sent = stats?.total_optimized_input_tokens || 0, saved = stats?.total_tokens_saved || 0
  const sentPct = raw ? Math.min(100, (sent / raw) * 100) : 100
  const stretch = useTween(Number(stats?.stretch_multiplier ?? 1), 1000)
  return (
    <section className="hero card rise" aria-label="Effective headroom">
      <div className="hero-top">
        <div>
          <p className="eyebrow">Effective headroom</p>
          <h1 className="hero-num mono">×{stretch.toFixed(1)}</h1>
          <p className="muted">Your context budget stretches this far after Gatekeep crushes the payload.</p>
        </div>
        <dl className="hero-facts">
          <div><dt>Saved</dt><dd className="mono"><Num value={saved} /></dd></div>
          <div><dt>Est. value</dt><dd className="mono"><Num value={stats?.estimated_usd_saved || 0} format={fmtUsd} /></dd></div>
          <div><dt>Avg latency</dt><dd className="mono"><Num value={stats?.avg_latency_ms || 0} format={(v) => `${Math.round(v)}ms`} /></dd></div>
        </dl>
      </div>
      <div className="flow" role="img" aria-label={raw ? `${fmt(sent)} sent of ${fmt(raw)} raw tokens` : 'No traffic yet'}>
        {raw > 0 ? (
          <div className="flow-bar">
            <div className="flow-sent" style={{ width: `${sentPct}%` }}><span className="flow-in mono">sent {fmt(sent)}</span></div>
            <div className="flow-crush" style={{ width: `${100 - sentPct}%` }}><span className="flow-stripes" /><span className="flow-in mono">crushed {fmt(saved)}</span></div>
          </div>
        ) : (
          // No traffic: an empty track with a hint, not a full bar that reads as "nothing saved".
          <div className="flow-bar flow-empty"><span className="flow-in">No traffic in this period · point a CLI at Gatekeep (Docs)</span></div>
        )}
        <div className="flow-legend mono">
          <span><i className="k-raw" />raw <b><Num value={raw} /></b></span>
          <span><i className="k-sent" />sent <b><Num value={sent} /></b></span>
          <span><i className="k-crush" />crushed <b><Num value={saved} /></b></span>
        </div>
      </div>
    </section>
  )
}

function Quotas({ usage, agent }) {
  if (!usage) return null
  const ant = usage.anthropic || (usage.five_hour_utilization != null ? usage : null), grok = usage.grok || null
  const cards = []
  if (ant && (agent === 'all' || agent === 'claude' || !grok)) {
    const st = (ant.five_hour_status || ant.status || 'allowed').toLowerCase()
    cards.push(
      <div className="quota card rise" key="ant">
        <div className="quota-head"><span>Claude subscription</span><span className={`pill ${st === 'allowed' ? 'ok' : 'bad'}`}>{st}</span></div>
        <div className="quota-rings">
          <div><Ring pct={ant.five_hour_utilization} color={loadColor(ant.five_hour_utilization ?? 0, 'var(--ok)')} /><b>5-hour window</b><span className="mono">resets {resetIn(ant.five_hour_reset)}</span></div>
          <div><Ring pct={ant.seven_day_utilization} color={loadColor(ant.seven_day_utilization ?? 0, 'var(--info)')} /><b>7-day limit</b><span className="mono">resets {resetIn(ant.seven_day_reset)}</span></div>
        </div>
        <span className="quota-foot mono">updated {ant.updated_at ? new Date(ant.updated_at).toLocaleTimeString() : '—'}</span>
      </div>,
    )
  }
  if (grok && (agent === 'all' || agent === 'grok' || !ant)) {
    const st = (grok.status || 'allowed').toLowerCase()
    cards.push(
      <div className="quota card rise" key="grok">
        <div className="quota-head"><span>Grok / xAI limits</span><span className={`pill ${st === 'allowed' ? 'ok' : 'bad'}`}>{st}</span></div>
        <div className="quota-rings">
          <div><Ring pct={grok.token_utilization} color={loadColor(grok.token_utilization ?? 0, 'var(--info)')} /><b>Tokens</b><span className="mono">{grok.remaining_tokens != null ? fmt(grok.remaining_tokens) : '—'} / {grok.limit_tokens != null ? fmt(grok.limit_tokens) : '—'} left</span></div>
          <div><Ring pct={grok.request_utilization} color={loadColor(grok.request_utilization ?? 0, 'var(--ok)')} /><b>Requests</b><span className="mono">{grok.remaining_requests != null ? fmt(grok.remaining_requests) : '—'} / {grok.limit_requests != null ? fmt(grok.limit_requests) : '—'} left</span></div>
        </div>
        <span className="quota-foot mono">updated {grok.updated_at ? new Date(grok.updated_at).toLocaleTimeString() : '—'}</span>
      </div>,
    )
  }
  return cards.length ? <div className="quotas">{cards}</div> : null
}

function Delta({ value }) {
  if (value == null || Number.isNaN(value)) return null
  if (Math.abs(value) >= 500) return <span className="delta">new this period</span>
  return <span className={`delta ${value > 0 ? 'up' : value < 0 ? 'down' : ''}`}>{value > 0 ? '▲' : value < 0 ? '▼' : '•'} {Math.abs(value).toFixed(1)}% vs prev</span>
}

function Agents({ stats, agent, onAgent }) {
  const list = stats?.savings_by_agent || []
  if (!list.length) return null
  return (
    <section className="card rise">
      <div className="sec-head"><h2>Saved by agent</h2>{agent !== 'all' && <button type="button" className="link" onClick={() => onAgent('all')}>Show all agents</button>}</div>
      <div className="agent-bar" role="group" aria-label="Savings by agent">
        {list.map((a) => a.pct_of_total_saved > 0 && (
          <button key={a.agent} type="button" title={`${a.label}: ${fmt(a.total_tokens_saved)} (${a.pct_of_total_saved}%)`} aria-label={`${a.label}: ${fmt(a.total_tokens_saved)} saved, ${a.pct_of_total_saved}%`} className={agent !== 'all' && agent !== a.agent ? 'dim' : ''} style={{ width: `${a.pct_of_total_saved}%`, background: agentColor(a.agent) }} onClick={() => onAgent(agent === a.agent ? 'all' : a.agent)} />
        ))}
      </div>
      <div className="agent-grid">
        {list.map((a, i) => (
          <button key={a.agent} type="button" className={`agent ${agent === a.agent ? 'on' : ''}`} style={{ '--i': i, '--c': agentColor(a.agent) }} onClick={() => onAgent(agent === a.agent ? 'all' : a.agent)}>
            <span className="agent-name"><i />{a.label}<em className="mono">{a.pct_of_total_saved}%</em></span>
            <b className="mono"><Num value={a.total_tokens_saved} /></b>
            <span className="mono muted">{a.total_requests} req · {a.avg_save_pct}% rate{a.estimated_usd_saved > 0 ? ` · ${fmtUsd(a.estimated_usd_saved)}` : ''}</span>
          </button>
        ))}
      </div>
    </section>
  )
}

function LiveFeed({ logs, onOpen, seen }) {
  const rows = logs.slice(0, 8)
  return (
    <section className="card rise feed">
      <div className="sec-head"><h2>Live requests</h2><span className="muted mono">{rows.length ? `latest ${ago(rows[0].timestamp)}` : 'waiting for traffic'}</span></div>
      {rows.length === 0 ? <div className="empty small"><span className="radar" /><span>Listening on the proxy…</span></div> : (
        <ul>
          {rows.map((r) => (
            <li key={r.id} className={seen.current.has(r.id) ? '' : 'new'}>
              <button type="button" onClick={() => onOpen(r)}>
                <i style={{ background: agentColor(r.agent) }} />
                <span className="f-model mono">{modelLabel(r)}</span>
                <span className="f-agent">{r.agent_label || r.agent || r.provider}</span>
                <span className="f-bar"><span style={{ width: `${Math.min(100, savePct(r))}%` }} /></span>
                <span className="f-pct mono">{savePct(r).toFixed(0)}%</span>
                <span className="f-time mono">{ago(r.timestamp)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function Techniques({ stats, filter, onPick }) {
  const list = Object.entries(stats?.technique_hits || {}).map(([name, v]) => ({ name, count: v.count || 0, saved: v.saved || 0 })).sort((a, b) => b.saved - a.saved).slice(0, 15)
  if (!list.length) return null
  const max = Math.max(...list.map((t) => t.saved), 1)
  return (
    <section className="card rise">
      <div className="sec-head"><h2>Crush techniques</h2><span className="muted">click to filter requests</span></div>
      <ul className="tech">
        {list.map((t, i) => (
          <li key={t.name}>
            <button type="button" className={filter === t.name ? 'on' : ''} onClick={() => onPick(t.name)} style={{ '--i': i }}>
              <span className="mono">{t.name}</span><span className="mono muted">{t.count} hits</span><b className="mono">{fmt(t.saved)}</b>
              <span className="tech-bar"><span style={{ width: `${(t.saved / max) * 100}%` }} /></span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}

function diffLines(a, b) {
  const A = (a || '').split('\n'), B = (b || '').split('\n'), n = A.length, m = B.length
  if (n * m > 4_000_000) return [{ t: 'same', s: '(payload too large to diff in the browser)' }]
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1))
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
  const out = []; let i = 0, j = 0
  while (i < n && j < m) { if (A[i] === B[j]) { out.push({ t: 'same', s: A[i] }); i++; j++ } else if (dp[i + 1][j] >= dp[i][j + 1]) out.push({ t: 'del', s: A[i++] }); else out.push({ t: 'add', s: B[j++] }) }
  while (i < n) out.push({ t: 'del', s: A[i++] })
  while (j < m) out.push({ t: 'add', s: B[j++] })
  return out
}

function Drawer({ log, onClose }) {
  const [tab, setTab] = useState('diff')
  const [closing, setClosing] = useState(false)
  const diff = useMemo(() => (log ? diffLines(log.original_payload_sample, log.optimized_payload_sample) : []), [log])
  useEffect(() => { setTab('diff'); setClosing(false) }, [log])
  if (!log) return null
  const close = () => { if (reduceMotion()) return onClose(); setClosing(true); setTimeout(onClose, 200) }
  const pct = savePct(log)
  return (
    <div className={`drawer-bg ${closing ? 'out' : ''}`} onClick={close}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Request ${log.id}`} onClick={(e) => e.stopPropagation()}>
        <header>
          <div><b className="mono">req/{log.id}</b><span className="muted">{fmtLogTime(log.timestamp)} · {log.agent_label || log.provider}</span></div>
          <button type="button" className="icon-btn" onClick={close} aria-label="Close">✕</button>
        </header>
        <div className="drawer-save">
          <Ring pct={pct} color="var(--ink)" size={64} />
          <dl>
            <div><dt>Raw</dt><dd className="mono">{fmt(log.original_input_tokens)}</dd></div>
            <div><dt>Sent</dt><dd className="mono">{fmt(log.optimized_input_tokens)}</dd></div>
            <div><dt>Saved</dt><dd className="mono">{fmt(log.input_tokens_saved)}</dd></div>
          </dl>
        </div>
        <dl className="kv">
          {[['Agent', log.agent_label || log.agent || '—'], ['Provider', log.provider || '—'], ['Model', modelLabel(log)], ['Technique', log.savings_category || '—']].map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{v}</dd></div>)}
        </dl>
        <Seg size="sm" label="Payload view" value={tab} onChange={setTab} items={[{ id: 'diff', label: 'Diff' }, { id: 'before', label: 'Before' }, { id: 'after', label: 'After' }]} />
        <pre className="code mono" key={tab}>
          {tab === 'before' && (log.original_payload_sample || '(empty)')}
          {tab === 'after' && (log.optimized_payload_sample || '(empty)')}
          {tab === 'diff' && diff.map((l, i) => <div key={i} className={`d-${l.t}`}>{l.t === 'add' ? '+ ' : l.t === 'del' ? '- ' : '  '}{l.s}</div>)}
        </pre>
      </aside>
    </div>
  )
}

function Settings({ settings, onToggle, onCleared }) {
  const [pricing, setPricing] = useState(null)
  const [busy, setBusy] = useState(null)
  const [msg, setMsg] = useState(null)
  const [err, setErr] = useState(null)
  const aggressive = !!settings.aggressive_enabled
  const load = useCallback(async () => {
    try { const r = await fetch('/api/prices'); if (!r.ok) throw new Error('offline'); setPricing(await r.json()); setErr(null) } catch (e) { setErr(e.message) }
  }, [])
  useEffect(() => { load() }, [load])
  const refresh = async () => {
    setBusy('prices'); setMsg(null)
    try { const r = await fetch('/api/prices/refresh', { method: 'POST' }); const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.detail || 'failed'); setMsg(`Updated ${d.models} models`); await load() } catch (e) { setErr(String(e.message || e)) } finally { setBusy(null) }
  }
  const clear = async () => {
    if (!window.confirm('Clear all logs and cache?')) return
    setBusy('clear'); setMsg(null)
    try { const r = await fetch('/api/data/clear', { method: 'POST' }); const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error('failed'); setMsg(`Cleared ${d.cleared_logs ?? 0} logs`); onCleared?.() } catch (e) { setErr(String(e.message || e)) } finally { setBusy(null) }
  }
  const models = pricing?.models ?? []
  return (
    <div className="stack">
      <section className="card rise">
        <div className="sec-head"><h2>Optimizations</h2><span className="muted">applies to new requests immediately</span></div>
        <ul className="toggles">
          {TOGGLES.map((t) => {
            const locked = aggressive && (t.key === 'history_pruning_enabled' || t.key === 'log_truncation_enabled')
            return (
              <li key={t.key}>
                <div><b>{t.label}</b><span className="muted">{locked ? 'Forced on by Aggressive' : t.hint}</span></div>
                <Switch checked={!!settings[t.key]} disabled={locked} label={t.label} onChange={(v) => onToggle(t.key, v)} />
              </li>
            )
          })}
        </ul>
      </section>
      <section className="card rise">
        <div className="sec-head"><h2>Data</h2><span className="muted mono">{pricing?.updated_at ? `prices ${pricing.updated_at} · ${pricing.count} in DB` : 'prices not fetched'}</span></div>
        <div className="row-btns">
          <button type="button" className="btn primary" disabled={busy === 'prices'} onClick={refresh}>{busy === 'prices' ? 'Updating…' : 'Update prices'}</button>
          <button type="button" className="btn danger" disabled={busy === 'clear'} onClick={clear}>{busy === 'clear' ? 'Clearing…' : 'Clear logs'}</button>
          {(msg || err) && <span className={err ? 'bad' : 'muted'}>{err || msg}</span>}
        </div>
      </section>
      <section className="card rise">
        <div className="sec-head"><h2>Model prices</h2><span className="muted">per million tokens</span></div>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th>Model</th><th className="num">Input</th><th className="num">Output</th><th>Source</th></tr></thead>
            <tbody>
              {models.length === 0 && <tr><td colSpan={4} className="empty-cell">No prices loaded</td></tr>}
              {models.map((m) => <tr key={m.display || m.model}><td className="mono">{m.display || m.model}</td><td className="num mono">{fmtRate(m.input_per_mtok)}</td><td className="num mono">{fmtRate(m.output_per_mtok)}</td><td className="muted">{m.source}</td></tr>)}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

/* ── App ───────────────────────────────────────────────────────────── */
const VIEWS = [{ id: 'overview', label: 'Overview' }, { id: 'requests', label: 'Requests' }, { id: 'settings', label: 'Settings' }, { id: 'docs', label: 'Docs' }]
const viewFromHash = () => { const h = location.hash.replace(/^#\/?/, ''); return VIEWS.some((v) => v.id === h) ? h : 'overview' }

export default function App() {
  const [view, setView] = useState(viewFromHash)
  const [range, setRange] = useState(() => localStorage.getItem('gk-range') || '24h')
  const [agent, setAgent] = useState('all')
  const [metricId, setMetricId] = useState('sent')
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)
  const [pulse, setPulse] = useState(0)
  const [logs, setLogs] = useState([])
  const [logsTotal, setLogsTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [noise, setNoise] = useState(false)
  const [technique, setTechnique] = useState(null)
  const [selected, setSelected] = useState(null)
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || 'dark')
  const seen = useRef(new Set())

  const q = `include_noise=${noise ? 1 : 0}&range=${encodeURIComponent(range)}${agent !== 'all' ? `&agent=${encodeURIComponent(agent)}` : ''}`
  const refresh = useCallback(async () => {
    try { const r = await fetch(`/api/stats?${q}`); if (!r.ok) throw new Error('offline'); setStats(await r.json()); setError(null); setPulse((p) => p + 1) } catch (e) { setError(e.message) }
  }, [q])
  const loadLogs = useCallback(async (offset = 0, append = false) => {
    if (append) setLoadingMore(true)
    try {
      const r = await fetch(`/api/logs?limit=${PAGE_SIZE}&offset=${offset}&${q}`); if (!r.ok) throw new Error('offline')
      const d = await r.json()
      setLogs((prev) => (append ? [...prev, ...d.logs] : d.logs)); setLogsTotal(d.total); setHasMore(d.has_more)
    } catch (e) { setError(e.message) } finally { setLoadingMore(false) }
  }, [q])

  useEffect(() => {
    const tick = () => { if (document.visibilityState === 'visible') { refresh(); loadLogs(0, false) } }
    tick(); const id = setInterval(tick, POLL)
    document.addEventListener('visibilitychange', tick)
    return () => { clearInterval(id); document.removeEventListener('visibilitychange', tick) }
  }, [refresh, loadLogs])
  // remember which rows were already shown, so only new ones animate in
  useEffect(() => { const t = setTimeout(() => logs.forEach((l) => seen.current.add(l.id)), 900); return () => clearTimeout(t) }, [logs])
  useEffect(() => { const on = () => setView(viewFromHash()); window.addEventListener('hashchange', on); return () => window.removeEventListener('hashchange', on) }, [])
  useEffect(() => { const on = (e) => { if (e.key === 'Escape') setSelected(null) }; window.addEventListener('keydown', on); return () => window.removeEventListener('keydown', on) }, [])
  useEffect(() => { try { localStorage.setItem('gk-range', range) } catch { /* private mode */ } }, [range])

  const go = (v) => { location.hash = `#/${v}`; setView(v) }
  const flipTheme = () => { const t = theme === 'light' ? 'dark' : 'light'; document.documentElement.dataset.theme = t; try { localStorage.setItem('gk-theme', t) } catch { /* private mode */ } setTheme(t) }
  const toggleSetting = async (key, next) => {
    setStats((s) => { if (!s) return s; const st = { ...s.settings, [key]: next ? 1 : 0 }; if (key === 'aggressive_enabled' && next) { st.history_pruning_enabled = 1; st.log_truncation_enabled = 1 } return { ...s, settings: st } })
    try { await fetch(`/api/settings/${key}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ value: next ? 1 : 0 }) }) } catch { refresh() }
  }

  const series = useMemo(() => {
    const ts = stats?.timeseries || []
    if (ts.some((b) => (b.original_input_tokens || 0) > 0 || (b.requests || 0) > 0)) return ts
    if (stats?.total_requests) return [{ t: new Date().toISOString(), original_input_tokens: stats.total_original_input_tokens || 0, input_tokens_saved: stats.total_tokens_saved || 0, optimized_input_tokens: stats.total_optimized_input_tokens || 0, requests: stats.total_requests || 0 }]
    return []
  }, [stats])
  const metric = METRICS.find((m) => m.id === metricId) || METRICS[0]
  const agents = stats?.available_agents || []
  const shownLogs = technique ? logs.filter((r) => (r.savings_category || '').includes(technique)) : logs
  const pickTechnique = (name) => { setTechnique((p) => (p === name ? null : name)); go('requests') }

  return (
    <div className="shell">
      <header className="top">
        <a className="brand" href="#/overview" onClick={(e) => { e.preventDefault(); go('overview') }}>
          <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" width="20" height="20" />Gatekeep
        </a>
        <Seg label="Views" value={view} onChange={go} items={VIEWS.map((v) => ({ ...v, label: v.id === 'requests' && (technique || agent !== 'all') ? `${v.label} •` : v.label }))} />
        <div className="top-right">
          <span className={`live ${error ? 'bad' : 'ok'}`} key={pulse} title={error ? 'Proxy unreachable' : 'Polling every 3s'}><i />{error ? 'offline' : 'live'}</span>
          <button type="button" className="icon-btn" onClick={flipTheme} aria-label="Toggle light or dark theme">{theme === 'light' ? '☾' : '☀'}</button>
        </div>
      </header>

      {(view === 'overview' || view === 'requests') && (
        <div className="filters">
          <Seg label="Time range" value={range} onChange={setRange} items={RANGES} size="sm" />
          {agents.length > 0 && <Seg label="Agent" value={agent} onChange={setAgent} size="sm" items={[{ id: 'all', label: 'All agents' }, ...agents.map((a) => ({ id: a.id, label: a.label, dot: agentColor(a.id) }))]} />}
        </div>
      )}

      <main className="page" key={view}>
        {view === 'overview' && !stats && error && (
          // API down and nothing loaded yet: say so, instead of zeros that read as "nothing saved".
          <section className="card rise offline-card" role="status">
            <p className="eyebrow">Effective headroom</p>
            <h1>Can't reach the Gatekeep API</h1>
            <p className="muted">The proxy did not answer ({error}). This page retries every {POLL / 1000} s; nothing is lost.</p>
          </section>
        )}
        {view === 'overview' && (stats || !error) && (
          <div className="stack">
            <Hero stats={stats} />
            <div className="ov-grid">
            <div className="ov-main">
            <div className="metrics">
              {METRICS.map((m, i) => (
                <button key={m.id} type="button" className={`metric ${metricId === m.id ? 'on' : ''}`} style={{ '--i': i }} onClick={() => setMetricId(m.id)} aria-pressed={metricId === m.id}>
                  <span className="m-label">{m.label}</span>
                  <b className="mono">{m.card(stats)}</b>
                  <Spark values={series.map(m.bucket)} />
                  {m.sub ? <span className="delta">{m.sub(stats)}</span> : <Delta value={m.delta?.(stats?.compare_previous)} />}
                </button>
              ))}
            </div>
            <section className="card rise">
              <div className="sec-head"><h2>{metric.label}</h2><span className="muted mono">{series.length} buckets · {stats?.granularity || ''}</span></div>
              <Chart series={series} metric={metric} />
            </section>
            <LiveFeed logs={logs} onOpen={setSelected} seen={seen} />
            </div>
            <aside className="ov-side" aria-label="Limits and breakdown">
              <Quotas usage={stats?.subscription_usage} agent={agent} />
              <Agents stats={stats} agent={agent} onAgent={setAgent} />
              <Techniques stats={stats} filter={technique} onPick={pickTechnique} />
            </aside>
            </div>
          </div>
        )}

        {view === 'requests' && (
          <section className="card rise">
            <div className="sec-head">
              <h2>Requests <span className="muted mono">{shownLogs.length}{technique || agent !== 'all' ? ' filtered' : ` / ${logsTotal}`}</span></h2>
              <div className="chips">
                {agent !== 'all' && <button type="button" className="chip" onClick={() => setAgent('all')}>{agents.find((a) => a.id === agent)?.label || agent} ✕</button>}
                {technique && <button type="button" className="chip" onClick={() => setTechnique(null)}>{technique} ✕</button>}
                <label className="check"><Switch checked={noise} label="Show telemetry" onChange={setNoise} />telemetry</label>
              </div>
            </div>
            <div className="tbl-wrap">
              <table className="tbl rows">
                <thead><tr><th>Time</th><th>Agent</th><th>Provider</th><th>Model</th><th className="num">Prompt</th><th className="num">Sent</th><th className="num">Saved</th><th className="num">Rate</th></tr></thead>
                <tbody>
                  {shownLogs.length === 0 && <tr><td colSpan={8} className="empty-cell">{technique || agent !== 'all' ? 'No requests for this filter' : 'No requests in this period'}</td></tr>}
                  {shownLogs.map((r, i) => (
                    <tr key={r.id} className={`${seen.current.has(r.id) ? '' : 'new'} ${selected?.id === r.id ? 'sel' : ''}`} style={{ '--i': Math.min(i, 20) }} tabIndex={0} onClick={() => setSelected(r)} onKeyDown={(e) => e.key === 'Enter' && setSelected(r)}>
                      <td className="mono muted">{fmtLogTime(r.timestamp)}</td>
                      <td><span className="tag"><i style={{ background: agentColor(r.agent) }} />{r.agent_label || r.agent || '—'}</span></td>
                      <td>{r.provider}</td>
                      <td className="mono muted">{modelLabel(r)}</td>
                      <td className="num mono">{fmt(r.original_input_tokens)}</td>
                      <td className="num mono">{fmt(r.optimized_input_tokens)}</td>
                      <td className="num mono">{fmt(r.input_tokens_saved)}</td>
                      <td className="num"><span className="rate"><span style={{ width: `${Math.min(100, savePct(r))}%` }} /><em className="mono">{savePct(r).toFixed(1)}</em></span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {hasMore && !technique && <button type="button" className="btn more" disabled={loadingMore} onClick={() => loadLogs(logs.length, true)}>{loadingMore ? 'Loading…' : 'Load more'}</button>}
          </section>
        )}

        {view === 'settings' && <Settings settings={stats?.settings ?? {}} onToggle={toggleSetting} onCleared={() => { refresh(); loadLogs(0, false) }} />}
        {view === 'docs' && <div className="docs rise"><WikiPage /></div>}
      </main>

      <footer className="foot">
        <a href="https://sealeria.com" target="_blank" rel="noopener noreferrer" aria-label="made by Sealeria"><span>made by</span><b>SEALERIA</b></a>
      </footer>
      <Drawer log={selected} onClose={() => setSelected(null)} />
    </div>
  )
}
