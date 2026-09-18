import { apiFetch } from './client'

export const checkHealth = () => apiFetch('/health')

export async function runAgent(prompt) {
  // This compact contract keeps the UI independent of the current demo adapter.
  const data = await apiFetch('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) })
  const trace = normaliseTrace(data?.trace || data?.steps)
  const runId = data?.run_id || data?.runId || data?.id
  if (trace.length || !runId) return { result: extractResult(data), trace }
  const traceResponse = await apiFetch(`/api/runs/${encodeURIComponent(runId)}/trace`)
  return { result: extractResult(data), trace: normaliseTrace(traceResponse?.trace || traceResponse?.steps || traceResponse) }
}

function extractResult(data) {
  if (typeof data === 'string') return data
  return data?.result ?? data?.response ?? data?.output ?? data?.final_response ?? 'The agent completed without returning a final response.'
}

function normaliseTrace(value) {
  const items = Array.isArray(value) ? value : []
  return items.map((item, index) => ({
    step: item?.step ?? item?.step_number ?? index + 1,
    action: item?.action ?? item?.tool_name ?? item?.tool ?? item?.name ?? 'Agent step',
    status: item?.status ?? 'completed',
    input: item?.input ?? item?.arguments ?? item?.payload,
    output: item?.output ?? item?.result ?? item?.message,
    timestamp: item?.timestamp ?? item?.created_at,
    duration: item?.duration ?? item?.duration_ms,
  }))
}
