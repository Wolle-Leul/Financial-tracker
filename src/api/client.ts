const base = () => (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

/** Signed token from POST /auth/login when the browser blocks cross-site cookies. */
const AUTH_TOKEN_KEY = 'ft_auth_token'

export function clearAuthToken(): void {
  sessionStorage.removeItem(AUTH_TOKEN_KEY)
}

async function apiFetch(path: string, init: RequestInit = {}) {
  const url = `${base()}${path.startsWith('/') ? path : `/${path}`}`
  const headers = new Headers(init.headers)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const token = sessionStorage.getItem(AUTH_TOKEN_KEY)
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  return fetch(url, {
    ...init,
    credentials: 'include',
    headers,
  })
}

export async function login(password: string): Promise<void> {
  clearAuthToken()
  const r = await apiFetch('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ password }),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || r.statusText)
  }
  const j = (await r.json()) as { token?: string }
  if (j.token) {
    sessionStorage.setItem(AUTH_TOKEN_KEY, j.token)
  }
}

export async function logout(): Promise<void> {
  clearAuthToken()
  await apiFetch('/auth/logout', { method: 'POST' })
}

export async function authMe(): Promise<boolean> {
  const r = await apiFetch('/auth/me')
  if (!r.ok) return false
  const j = (await r.json()) as { authenticated?: boolean }
  return Boolean(j.authenticated)
}

export type KpiItem = { title: string; value: string; subtitle: string }

export type DashboardResponse = {
  kpis: KpiItem[]
  filter_categories: string[]
  filter_subcategories: string[]
  days_till_next_salary: number
  days_left_for_infy_label: string
  due_salary_date: string
  groceries_amount: number
  net_of_net: number
  needs_percent: number
  target_vs_actual_percent: number
  salary_prev_month: string
  calendar_year: number
  calendar_month: number
  calendar_html: string
  sankey_plotly: { data?: unknown[]; layout?: Record<string, unknown> }
  to_pay_rows: Array<{
    Category: string
    SubCategory: string
    Deadline?: string | null
    PlannedAmount: number
    ActualAmount: number
    Variance: number
  }>
  to_pay_empty_reason?: string | null
  import_quality_value: string
  import_quality_sub: string
}

function buildQuery(params: Record<string, string | number | string[] | undefined>) {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined) continue
    if (Array.isArray(v)) {
      for (const item of v) q.append(k, item)
    } else {
      q.set(k, String(v))
    }
  }
  const s = q.toString()
  return s ? `?${s}` : ''
}

export async function fetchDashboard(opts: {
  year: number
  month: number
  categories?: string[]
  subcategories?: string[]
}): Promise<DashboardResponse> {
  const q = buildQuery({
    year: opts.year,
    month: opts.month,
    categories: opts.categories,
    subcategories: opts.subcategories,
  })
  const r = await apiFetch(`/api/dashboard${q}`)
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || r.statusText)
  }
  return r.json() as Promise<DashboardResponse>
}

export async function importPdf(file: File): Promise<{
  import_id: number
  rows_added: number
  parse_warnings_count: number
}> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await apiFetch('/api/imports/pdf', { method: 'POST', body: fd })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || r.statusText)
  }
  return r.json() as Promise<{
    import_id: number
    rows_added: number
    parse_warnings_count: number
  }>
}

export type UncategorizedResponse = {
  transactions: Array<{ id: number; label: string }>
  subcategories: Array<{ id: number; label: string }>
}

export async function fetchUncategorized(importId: number): Promise<UncategorizedResponse> {
  const r = await apiFetch(`/api/imports/${importId}/uncategorized`)
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) throw new Error(await r.text())
  return r.json() as Promise<UncategorizedResponse>
}

export async function mapTransaction(transactionId: number, subcategoryId: number): Promise<void> {
  const r = await apiFetch(`/api/transactions/${transactionId}`, {
    method: 'PATCH',
    body: JSON.stringify({ subcategory_id: subcategoryId }),
  })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || r.statusText)
  }
}
