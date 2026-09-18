function display(value) { if (value == null || value === '') return '—'; return typeof value === 'string' ? value : JSON.stringify(value) }
function TraceStep({ step, active = false }) {
  const status = String(step.status || 'completed').toLowerCase()
  const statusClass = ['success', 'completed', 'error', 'failed', 'running', 'started'].includes(status) ? status : 'default'
  const providerEvent = String(step.action || '').startsWith('LLM_')
  return <article className={`trace-step${providerEvent ? ' provider-event' : ''}${active ? ' active-step' : ''}`}><div className="trace-step-header"><span className="step-number">{step.step}</span><span className="action-name">{step.action}</span>{active && <span className="live-dot" aria-label="Currently executing" />}{providerEvent && <span className="infra-badge">provider</span>}<span className={`status-pill status-${statusClass}`}>{status}</span></div>{(step.timestamp || step.duration != null) && <p className="trace-meta">{[step.timestamp, step.duration != null && `Duration: ${Number(step.duration).toFixed(1)} ms`].filter(Boolean).join(' · ')}</p>}<div className="trace-data"><div><strong>{providerEvent ? 'Details' : 'Input'}</strong>{display(step.input)}</div><div><strong>{providerEvent ? 'Event' : 'Output'}</strong>{display(step.output)}</div></div></article>
}
export default TraceStep
