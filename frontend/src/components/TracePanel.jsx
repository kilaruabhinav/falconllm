import TraceStep from './TraceStep'
function TracePanel({ trace, isRunning, activity }) {
  return <section className="panel trace-panel"><div className="panel-heading"><h2>Execution trace</h2><span>{isRunning ? activity : trace.length ? `${trace.length} step${trace.length === 1 ? '' : 's'}` : 'Live when available'}</span></div>{trace.length ? <div className="trace-list">{trace.map((step, index) => <TraceStep key={step.id || `${step.step}-${index}`} step={step} active={isRunning && index === trace.length - 1} />)}</div> : <p className="empty-state">{isRunning ? 'Connecting to the live execution trace…' : 'Run an agent task to inspect its tools, actions, and outputs.'}</p>}</section>
}
export default TracePanel
