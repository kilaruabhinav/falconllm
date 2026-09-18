function display(value) { if (value == null || value === '') return '—'; return typeof value === 'string' ? value : JSON.stringify(value) }
function TraceStep({ step }) {
  const status = String(step.status || 'completed').toLowerCase()
  const statusClass = ['success', 'completed', 'error', 'failed', 'running'].includes(status) ? status : 'default'
  return <article className="trace-step"><div className="trace-step-header"><span className="step-number">{step.step}</span><span className="action-name">{step.action}</span><span className={`status-pill status-${statusClass}`}>{status}</span></div>{(step.timestamp || step.duration) && <p className="trace-meta">{[step.timestamp, step.duration && `Duration: ${step.duration}`].filter(Boolean).join(' · ')}</p>}<div className="trace-data"><div><strong>Input</strong>{display(step.input)}</div><div><strong>Output</strong>{display(step.output)}</div></div></article>
}
export default TraceStep
