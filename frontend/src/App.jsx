import { useCallback, useEffect, useRef, useState } from 'react'
import { checkHealth, getRun, getRunTrace, getRuns, normaliseTrace, openRunStream, startAgentRun } from './api/agent'
import { activityFor, mergeTraceSteps } from './api/stream'
import AgentInput from './components/AgentInput'
import AgentResult from './components/AgentResult'
import ConnectionStatus from './components/ConnectionStatus'
import Header from './components/Header'
import TracePanel from './components/TracePanel'
import './App.css'
import './History.css'
import './LiveTrace.css'
import './ProviderTrace.css'

const ACTIVE_RUN_KEY = 'falconllm-active-run'
const TERMINAL = new Set(['RUN_COMPLETED', 'RUN_FAILED', 'RUN_CANCELLED'])

function App() {
  const [connection, setConnection] = useState('checking')
  const [isRunning, setIsRunning] = useState(false)
  const [activity, setActivity] = useState('')
  const [result, setResult] = useState(null)
  const [trace, setTrace] = useState([])
  const [error, setError] = useState('')
  const [runs, setRuns] = useState([])
  const streamRef = useRef(null)

  const refreshRuns = useCallback(async () => {
    const data = await getRuns()
    setRuns(data.runs || [])
  }, [])

  const refreshHealth = useCallback(async () => {
    setConnection('checking')
    try {
      await checkHealth()
      setConnection('online')
      await refreshRuns()
    } catch {
      setConnection('offline')
    }
  }, [refreshRuns])

  const finishRun = useCallback(async (runId, terminalStep) => {
    streamRef.current?.close()
    streamRef.current = null
    window.localStorage.removeItem(ACTIVE_RUN_KEY)
    setIsRunning(false)
    setActivity('')
    if (terminalStep.action === 'RUN_COMPLETED') {
      const answer = terminalStep.metadata?.final_answer
      if (answer) setResult(answer)
      else {
        try { setResult((await getRun(runId)).final_answer) } catch { /* completion is already visible */ }
      }
    } else {
      setError(String(terminalStep.output || terminalStep.action))
    }
    try { await refreshRuns() } catch { /* history is non-critical */ }
  }, [refreshRuns])

  const connectToRun = useCallback(async (runId) => {
    streamRef.current?.close()
    let persisted = []
    try {
      const response = await getRunTrace(runId)
      persisted = normaliseTrace(response.trace || response.steps || [])
      setTrace((current) => mergeTraceSteps(current, persisted))
    } catch { /* the task may not have persisted RUN_STARTED yet */ }

    const last = persisted.at(-1)
    if (last && TERMINAL.has(last.action)) {
      await finishRun(runId, last)
      return
    }
    streamRef.current = openRunStream(runId, last?.step || 0, {
      onTrace: (step) => {
        setTrace((current) => mergeTraceSteps(current, [step]))
        setActivity(activityFor(step))
        if (step.action === 'FINAL') setResult(String(step.output || ''))
        if (TERMINAL.has(step.action)) void finishRun(runId, step)
      },
      onDisconnect: (source) => {
        if (source.readyState !== EventSource.OPEN) setActivity('Reconnecting live trace…')
      },
      onFatal: (streamError) => {
        window.localStorage.removeItem(ACTIVE_RUN_KEY)
        setError(streamError.message)
        setIsRunning(false)
        setActivity('')
      },
    })
  }, [finishRun])

  useEffect(() => {
    const timer = window.setTimeout(() => { void refreshHealth() }, 0)
    return () => window.clearTimeout(timer)
  }, [refreshHealth])

  useEffect(() => {
    const activeRun = window.localStorage.getItem(ACTIVE_RUN_KEY)
    const timer = activeRun ? window.setTimeout(() => {
      setIsRunning(true)
      setActivity('Reconnecting live trace…')
      void connectToRun(activeRun)
    }, 0) : null
    return () => {
      if (timer !== null) window.clearTimeout(timer)
      streamRef.current?.close()
    }
  }, [connectToRun])

  async function handleRun(prompt) {
    streamRef.current?.close()
    setIsRunning(true)
    setActivity('Starting agent…')
    setError('')
    setResult(null)
    setTrace([])
    try {
      const response = await startAgentRun(prompt)
      window.localStorage.setItem(ACTIVE_RUN_KEY, response.run_id)
      setConnection('online')
      await connectToRun(response.run_id)
    } catch (requestError) {
      setError(requestError.message || 'The agent request could not be completed.')
      setIsRunning(false)
      setActivity('')
      if ([404, 405].includes(requestError.status)) setConnection('online')
    }
  }

  return <div className="app-shell"><Header status={<ConnectionStatus status={connection} onRefresh={refreshHealth} />} /><main className="workspace"><section className="intro" aria-labelledby="page-title"><p className="eyebrow">Agent workspace</p><h1 id="page-title">Give FalconLLM a task.</h1><p className="intro-copy">Submit a prompt, then follow the agent&apos;s operational plans, tool calls, recovery events, and final answer.</p></section><AgentInput isRunning={isRunning} onSubmit={handleRun} /><AgentResult result={result} error={error} isRunning={isRunning} activity={activity} /><TracePanel trace={trace} isRunning={isRunning} activity={activity} /><section className="panel history-panel"><div className="panel-heading"><h2>Run history</h2><span>{runs.length} recent</span></div>{runs.length ? <div className="history-list">{runs.map((run) => <article key={run.run_id}><strong>{run.user_query}</strong><span>{run.status}</span><p>{run.final_answer || run.error_json || (run.status === 'RUNNING' ? 'Currently running…' : 'No final answer')}</p></article>)}</div> : <p className="empty-state">Completed runs will appear here.</p>}</section></main></div>
}

export default App
