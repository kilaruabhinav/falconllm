import TraceStep from './TraceStep'
function TracePanel({ trace, isRunning }) {
  return <section className="panel trace-panel"><div className="panel-heading"><h2>Execution trace</h2><span>{trace.length ? `${trace.length} step${trace.length === 1 ? '' : 's'}` : 'Live when available'}</span></div>{trace.length ? <div className="trace-list">{trace.map((step, index) => <TraceStep key={`${step.step}-${index}`} step={step} />)}</div> : <p className="empty-state">{isRunning ? 'Trace events will appear as the agent reports them.' : 'Run an agent task to inspect its tools, actions, and outputs.'}</p>}</section>
}
export default TracePanel
