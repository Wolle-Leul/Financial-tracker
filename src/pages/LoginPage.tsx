import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { authMe, login } from '../api/client'

export default function LoginPage() {
  const nav = useNavigate()
  const qc = useQueryClient()
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const { tokenReceived } = await login(password)
      const sessionOk = await authMe()
      if (!sessionOk) {
        if (!tokenReceived) {
          throw new Error(
            'The API did not return a login token. On Render: set Start Command to PYTHONPATH=. python -m uvicorn run_api:app --host 0.0.0.0 --port $PORT and Build to pip install -r requirements.txt, then redeploy. Test POST /auth/login in /docs — body should include "token".',
          )
        }
        throw new Error(
          'Login token was ignored by the server. On Render set SESSION_SECRET to a long random string (and keep it stable), redeploy API, clear site data, try again.',
        )
      }
      qc.removeQueries({ queryKey: ['dashboard'] })
      nav('/', { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-wrap">
      <div className="login-card">
        <h1>Finance Tracker</h1>
        <p className="muted">Sign in to continue</p>
        <form onSubmit={onSubmit}>
          <label className="label" htmlFor="pw">
            Password
          </label>
          <input
            id="pw"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="input"
            required
          />
          {error ? <p className="error">{error}</p> : null}
          <button type="submit" className="btn primary" disabled={loading}>
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  )
}
