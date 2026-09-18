import { apiFetch, apiUrl } from './client'
import { normaliseTrace, normaliseTraceStep } from './stream'

export const checkHealth = () => apiFetch('/health')
export const getRuns = () => apiFetch('/api/runs')
export const getRun = (runId) => apiFetch(`/api/runs/${encodeURIComponent(runId)}`)
export const getRunTrace = (runId) => apiFetch(`/api/runs/${encodeURIComponent(runId)}/trace`)

export async function startAgentRun(prompt) {
  return apiFetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  })
}

export function openRunStream(runId, afterSequence, handlers) {
  const query = afterSequence ? `?after=${afterSequence}` : ''
  const source = new EventSource(apiUrl(`/api/runs/${encodeURIComponent(runId)}/stream${query}`))
  source.addEventListener('trace', (message) => {
    try {
      handlers.onTrace(normaliseTraceStep(JSON.parse(message.data)))
    } catch {
      handlers.onFatal?.(new Error('The live trace contained an invalid event.'))
      source.close()
    }
  })
  source.onerror = () => handlers.onDisconnect?.(source)
  return source
}

export { normaliseTrace }
