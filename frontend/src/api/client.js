const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export const apiUrl = (path) => `${API_BASE_URL}${path}`

export class ApiError extends Error {
  constructor(message, status) { super(message); this.name = 'ApiError'; this.status = status }
}

export async function apiFetch(path, options = {}) {
  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { headers: { Accept: 'application/json', ...options.headers }, ...options })
  } catch {
    throw new ApiError(`Unable to reach the FalconLLM API at ${API_BASE_URL}. Start the backend and check VITE_API_BASE_URL.`)
  }
  const contentType = response.headers.get('content-type') || ''
  let body
  try {
    body = contentType.includes('application/json') ? await response.json() : await response.text()
  } catch {
    throw new ApiError('The FalconLLM API returned an unreadable response.', response.status)
  }
  if (!response.ok) {
    const detail = typeof body === 'object' ? body.detail : body
    throw new ApiError(detail || `Request failed (${response.status})`, response.status)
  }
  return body
}
