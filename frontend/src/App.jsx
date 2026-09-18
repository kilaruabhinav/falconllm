import { useCallback, useEffect, useState } from 'react'
import { checkHealth, runAgent } from './api/agent'
import AgentInput from './components/AgentInput'
import AgentResult from './components/AgentResult'
import ConnectionStatus from './components/ConnectionStatus'
import Header from './components/Header'
import TracePanel from './components/TracePanel'
import './App.css'

function App() {
  const [connection, setConnection] = useState('checking')
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [trace, setTrace] = useState([])
  const [error, setError] = useState('')
  const refreshHealth = useCallback(async () => {
    setConnection('checking')
    try { await checkHealth(); setConnection('online') } catch { setConnection('offline') }
  }, [])
  useEffect(() => {
    const timer = window.setTimeout(() => { void refreshHealth() }, 0)
    return () => window.clearTimeout(timer)
  }, [refreshHealth])
  async function handleRun(prompt) {
    setIsRunning(true); setError(''); setResult(null); setTrace([])
    try { const response = await runAgent(prompt); setResult(response.result); setTrace(response.trace); setConnection('online') }
    catch (requestError) { setError(requestError.message || 'The agent request could not be completed.'); if ([404, 405].includes(requestError.status)) setConnection('online') }
    finally { setIsRunning(false) }
  }
  return <div className="app-shell"><Header status={<ConnectionStatus status={connection} onRefresh={refreshHealth} />} /><main className="workspace"><section className="intro" aria-labelledby="page-title"><p className="eyebrow">Agent workspace</p><h1 id="page-title">Give FalconLLM a task.</h1><p className="intro-copy">Submit a prompt, then follow the agent&apos;s response and execution trace in one place.</p></section><AgentInput isRunning={isRunning} onSubmit={handleRun} /><AgentResult result={result} error={error} isRunning={isRunning} /><TracePanel trace={trace} isRunning={isRunning} /></main></div>
}
export default App
