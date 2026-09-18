import assert from 'node:assert/strict'
import test from 'node:test'

import { activityFor, mergeTraceSteps, normaliseTraceStep } from './stream.js'

test('normalises streamed trace payloads', () => {
  const step = normaliseTraceStep({ step_id: 'a', sequence: 2, step_type: 'PLAN', content: 'Plan' })
  assert.equal(step.id, 'a')
  assert.equal(step.step, 2)
  assert.equal(step.output, 'Plan')
})

test('reconnect merge removes duplicate persisted and streamed events', () => {
  const first = normaliseTraceStep({ step_id: 'a', sequence: 1, step_type: 'RUN_STARTED' })
  const updated = { ...first, status: 'SUCCESS' }
  const second = normaliseTraceStep({ step_id: 'b', sequence: 2, step_type: 'PLAN' })
  const merged = mergeTraceSteps([first], [updated, second])
  assert.deepEqual(merged.map((step) => step.id), ['a', 'b'])
  assert.equal(merged[0].status, 'SUCCESS')
})

test('maps live tool activity', () => {
  assert.equal(activityFor({ action: 'TOOL_CALL', tool: 'search' }), 'Searching web…')
  assert.equal(activityFor({ action: 'LLM_FALLBACK' }), 'Switching LLM…')
})
