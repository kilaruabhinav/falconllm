import { useState } from 'react'
function AgentInput({ isRunning, onSubmit }) {
  const [prompt, setPrompt] = useState('')
  function submit(event) { event.preventDefault(); if (prompt.trim()) onSubmit(prompt.trim()) }
  return <form className="panel input-panel" onSubmit={submit}><label className="field-label" htmlFor="agent-prompt">Your task</label><textarea className="prompt-input" id="agent-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Ask FalconLLM to research, plan, analyze, or use a tool…" disabled={isRunning} /><div className="input-footer"><span>Enter a clear task to begin.</span><button className="run-button" type="submit" disabled={isRunning || !prompt.trim()}>{isRunning ? 'Running agent…' : 'Run Agent'}</button></div></form>
}
export default AgentInput
