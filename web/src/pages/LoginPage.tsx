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
      await login(password)
      const sessionOk = await authMe()
      if (!sessionOk) {
        throw new Error(
          'Password accepted but the session cookie was not saved. On Render set SESSION_SAME_SITE=none (HTTPS), and set CORS_ORIGINS to this site’s exact origin (https://…, no trailing slash).',
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
