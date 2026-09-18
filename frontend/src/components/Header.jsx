function Header({ status }) {
  return <header className="topbar"><a className="brand" href="/"><span className="brand-mark" aria-hidden="true" />FalconLLM</a>{status}</header>
}
export default Header
