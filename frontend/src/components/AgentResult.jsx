function AgentResult({ result, error, isRunning, activity }) {
  return <section className="panel result-panel" aria-live="polite"><div className="panel-heading"><h2>Final response</h2><span className={`run-state ${isRunning ? 'active' : ''}`}>{isRunning ? 'RUNNING' : result ? 'COMPLETED' : error ? 'FAILED' : 'IDLE'}</span></div>{error ? <p className="error-state">{error}</p> : result ? <p className="result-content">{String(result)}</p> : isRunning ? <p className="loading-line"><span className="spinner" />{activity || 'Waiting for the agent…'}</p> : <p className="empty-state">Your agent&apos;s final response will appear here.</p>}</section>
}
export default AgentResult
