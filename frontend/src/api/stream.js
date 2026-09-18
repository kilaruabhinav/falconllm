export function normaliseTraceStep(item, index = 0) {
  return {
    id: item?.step_id || `${item?.run_id || 'run'}-${item?.sequence ?? item?.step_number ?? index + 1}`,
    step: item?.sequence ?? item?.step ?? item?.step_number ?? index + 1,
    action: item?.step_type ?? item?.type ?? item?.action ?? item?.tool_name ?? item?.tool ?? item?.name ?? 'Agent step',
    status: item?.status ?? 'completed',
    input: item?.input ?? item?.arguments ?? item?.metadata ?? item?.payload,
    output: item?.output ?? item?.observation ?? item?.result ?? item?.error ?? item?.content ?? item?.message,
    timestamp: item?.timestamp ?? item?.created_at,
    duration: item?.duration ?? item?.duration_ms,
    metadata: item?.metadata ?? {},
    tool: item?.tool_name ?? item?.tool,
  }
}

export function normaliseTrace(value) {
  return (Array.isArray(value) ? value : []).map(normaliseTraceStep)
}

export function mergeTraceSteps(current, incoming) {
  const merged = new Map(current.map((step) => [step.id || `sequence-${step.step}`, step]))
  for (const step of incoming) merged.set(step.id || `sequence-${step.step}`, step)
  return [...merged.values()].sort((a, b) => a.step - b.step)
}

export function activityFor(step) {
  if (!step) return 'Starting agent…'
  if (step.action === 'LLM_FALLBACK') return 'Switching LLM…'
  if (step.action === 'LLM_PROVIDER_SELECTED') return 'Generating next action…'
  if (step.action === 'RECOVERY' || step.action.endsWith('_ERROR')) return 'Recovering…'
  if (step.action === 'TOOL_CALL') {
    if (step.tool === 'search') return 'Searching web…'
    if (step.tool === 'calculator') return 'Calculating…'
    if (step.tool === 'file_reader') return 'Reading file…'
    return 'Running tool…'
  }
  if (step.action === 'PLAN') return 'Planning next step…'
  if (step.action === 'FINAL') return 'Finalizing answer…'
  return 'Agent is running…'
}
