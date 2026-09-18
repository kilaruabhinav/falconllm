import { useCallback, useEffect, useState } from 'react'
import { checkHealth, getRuns, runAgent } from './api/agent'
import AgentInput from './components/AgentInput'
import AgentResult from './components/AgentResult'
import ConnectionStatus from './components/ConnectionStatus'
import Header from './components/Header'
import TracePanel from './components/TracePanel'
import './App.css'
import './History.css'

function App() {
  const [connection, setConnection] = useState('checking')
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [trace, setTrace] = useState([])
  const [error, setError] = useState('')
  const [runs, setRuns] = useState([])
  const refreshHealth = useCallback(async () => {
    setConnection('checking')
    try { await checkHealth(); setConnection('online'); const data = await getRuns(); setRuns(data.runs || []) } catch { setConnection('offline') }
  }, [])
  useEffect(() => {
    const timer = window.setTimeout(() => { void refreshHealth() }, 0)
    return () => window.clearTimeout(timer)
  }, [refreshHealth])
  async function handleRun(prompt) {
    setIsRunning(true); setError(''); setResult(null); setTrace([])
    try { const response = await runAgent(prompt); setResult(response.result); setTrace(response.trace); setConnection('online'); const data = await getRuns(); setRuns(data.runs || []) }
    catch (requestError) { setError(requestError.message || 'The agent request could not be completed.'); if ([404, 405].includes(requestError.status)) setConnection('online') }
    finally { setIsRunning(false) }
  }
  return <div className="app-shell"><Header status={<ConnectionStatus status={connection} onRefresh={refreshHealth} />} /><main className="workspace"><section className="intro" aria-labelledby="page-title"><p className="eyebrow">Agent workspace</p><h1 id="page-title">Give FalconLLM a task.</h1><p className="intro-copy">Submit a prompt, then follow the agent&apos;s operational plans, tool calls, recovery events, and final answer.</p></section><AgentInput isRunning={isRunning} onSubmit={handleRun} /><AgentResult result={result} error={error} isRunning={isRunning} /><TracePanel trace={trace} isRunning={isRunning} /><section className="panel history-panel"><div className="panel-heading"><h2>Run history</h2><span>{runs.length} recent</span></div>{runs.length ? <div className="history-list">{runs.map((run) => <article key={run.run_id}><strong>{run.user_query}</strong><span>{run.status}</span><p>{run.final_answer || run.error_json || 'No final answer'}</p></article>)}</div> : <p className="empty-state">Completed runs will appear here.</p>}</section></main></div>
}
export default App
