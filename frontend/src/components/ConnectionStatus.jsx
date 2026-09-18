const labels = { checking: 'Checking backend', online: 'Backend connected', offline: 'Backend unavailable' }
function ConnectionStatus({ status, onRefresh }) {
  return <div className="connection" aria-live="polite"><span className={`connection-dot ${status}`} aria-hidden="true" />{labels[status]}<button className="refresh-button" type="button" onClick={onRefresh}>Retry</button></div>
}
export default ConnectionStatus
