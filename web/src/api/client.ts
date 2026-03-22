const base = () => (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

/** Signed token from POST /auth/login when the browser blocks cross-site cookies. */
const AUTH_TOKEN_KEY = 'ft_auth_token'

export function clearAuthToken(): void {
  sessionStorage.removeItem(AUTH_TOKEN_KEY)
}

/** FastAPI often returns { detail: string | array }; static hosts return HTML — explain both. */
export async function readApiErrorMessage(r: Response): Promise<string> {
  const text = await r.text()
  try {
    const j = JSON.parse(text) as { detail?: unknown }
    const d = j.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) {
      return d
        .map((x) =>
          typeof x === 'object' && x !== null && 'msg' in x ? String((x as { msg: string }).msg) : JSON.stringify(x),
        )
        .join('; ')
    }
  } catch {
    /* not JSON */
  }
  if (text.includes('<!DOCTYPE') || text.includes('<html')) {
    return `HTTP ${r.status}: server returned HTML (not JSON). In production set VITE_API_BASE_URL to your API origin (e.g. https://….onrender.com).`
  }
  return text.trim().slice(0, 400) || `${r.status} ${r.statusText}`
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
  try {
    return await fetch(url, {
      ...init,
      credentials: 'include',
      headers,
    })
  } catch (e) {
    const hint =
      base() === '' && import.meta.env.PROD
        ? ' VITE_API_BASE_URL is empty — the SPA is calling the wrong host for /api.'
        : ''
    if (e instanceof TypeError) {
      throw new Error(`Cannot reach API (${url || path}). ${hint}`.trim())
    }
    throw e
  }
}

/** Set when POST /auth/login JSON includes `token` (Bearer auth for cross-origin SPA). */
export type LoginResult = { tokenReceived: boolean }

export async function login(password: string): Promise<LoginResult> {
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
    return { tokenReceived: true }
  }
  return { tokenReceived: false }
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

export type KpiItem = { title: string; value: string; subtitle: string; kind?: string }

export type DashboardResponse = {
  kpis: KpiItem[]
  filter_categories: string[]
  filter_subcategories: string[]
  days_till_next_salary: number
  salary_countdown_label: string
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
  expected_income_net?: number | null
  income_variance_vs_expected_percent?: number | null
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
    throw new Error(await readApiErrorMessage(r))
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
    throw new Error(await readApiErrorMessage(r))
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
  if (!r.ok) throw new Error(await readApiErrorMessage(r))
  return r.json() as Promise<UncategorizedResponse>
}

export async function mapTransaction(transactionId: number, subcategoryId: number): Promise<void> {
  const r = await apiFetch(`/api/transactions/${transactionId}`, {
    method: 'PATCH',
    body: JSON.stringify({ subcategory_id: subcategoryId }),
  })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    throw new Error(await readApiErrorMessage(r))
  }
}

export type SalaryRuleSettings = {
  salary_day_of_month: number
  holiday_country: string
  target_ratio: number
  budget_strategy: string
}

export async function fetchSalaryRuleSettings(): Promise<SalaryRuleSettings> {
  const r = await apiFetch('/api/settings/salary-rule')
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) throw new Error(await readApiErrorMessage(r))
  return r.json() as Promise<SalaryRuleSettings>
}

export async function patchSalaryRuleSettings(body: Partial<SalaryRuleSettings>): Promise<SalaryRuleSettings> {
  const r = await apiFetch('/api/settings/salary-rule', { method: 'PATCH', body: JSON.stringify(body) })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    throw new Error(await readApiErrorMessage(r))
  }
  return r.json() as Promise<SalaryRuleSettings>
}

/** Persist salary rule + all income rows + all recurring rows in one DB transaction (used for global Save). */
export type SettingsSyncBody = {
  salary_day_of_month: number
  target_ratio: number
  budget_strategy: string
  income_rows: Array<{
    id: number
    label?: string
    net_amount?: number | null
    gross_amount?: number | null
    salary_day_of_month?: number | null
  }>
  recurring_rows: Array<{
    subcategory_id: number
    planned_amount?: number | null
    planned_deadline_day?: number | null
  }>
}

export type SettingsSyncResponse = {
  salary_rule: SalaryRuleSettings
  income_rows_updated: number
  recurring_rows_updated: number
}

export async function postSettingsSync(body: SettingsSyncBody): Promise<SettingsSyncResponse> {
  const r = await apiFetch('/api/settings/sync', { method: 'POST', body: JSON.stringify(body) })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    throw new Error(await readApiErrorMessage(r))
  }
  return r.json() as Promise<SettingsSyncResponse>
}

export type IncomeSource = {
  id: number
  label: string
  employer_name?: string | null
  contract_type: string
  gross_amount?: number | null
  net_amount?: number | null
  use_net_only: boolean
  sort_order: number
  salary_day_of_month?: number | null
}

export async function fetchIncomeSources(): Promise<IncomeSource[]> {
  const r = await apiFetch('/api/income-sources')
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) throw new Error(await readApiErrorMessage(r))
  return r.json() as Promise<IncomeSource[]>
}

export async function createIncomeSource(body: Omit<IncomeSource, 'id'>): Promise<IncomeSource> {
  const r = await apiFetch('/api/income-sources', { method: 'POST', body: JSON.stringify(body) })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    throw new Error(await readApiErrorMessage(r))
  }
  return r.json() as Promise<IncomeSource>
}

export async function patchIncomeSource(id: number, body: Partial<IncomeSource>): Promise<IncomeSource> {
  const r = await apiFetch(`/api/income-sources/${id}`, { method: 'PATCH', body: JSON.stringify(body) })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || r.statusText)
  }
  return r.json() as Promise<IncomeSource>
}

export async function deleteIncomeSource(id: number): Promise<void> {
  const r = await apiFetch(`/api/income-sources/${id}`, { method: 'DELETE' })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) throw new Error(await readApiErrorMessage(r))
}

export type RecurringExpenseRow = {
  id: number
  category_name: string
  name: string
  planned_amount?: number | null
  planned_deadline_day?: number | null
}

export async function fetchRecurringExpenses(): Promise<RecurringExpenseRow[]> {
  const r = await apiFetch('/api/recurring-expenses')
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) throw new Error(await readApiErrorMessage(r))
  return r.json() as Promise<RecurringExpenseRow[]>
}

export async function patchRecurringExpense(
  id: number,
  body: { planned_amount?: number | null; planned_deadline_day?: number | null },
): Promise<RecurringExpenseRow> {
  const r = await apiFetch(`/api/recurring-expenses/${id}`, { method: 'PATCH', body: JSON.stringify(body) })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || r.statusText)
  }
  return r.json() as Promise<RecurringExpenseRow>
}

export type CalculateNetResponse = { net: number; notes: string }

export async function calculateNet(gross: number, contractType: string): Promise<CalculateNetResponse> {
  const r = await apiFetch('/api/calculate-net', {
    method: 'POST',
    body: JSON.stringify({ gross, contract_type: contractType }),
  })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || r.statusText)
  }
  return r.json() as Promise<CalculateNetResponse>
}

export type TrendPoint = { month: string; income: number; expenses: number; net: number }

export type TrendsResponse = { months: TrendPoint[] }

export async function fetchTrends(months = 12): Promise<TrendsResponse> {
  const r = await apiFetch(`/api/analytics/trends?months=${months}`)
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) throw new Error(await readApiErrorMessage(r))
  return r.json() as Promise<TrendsResponse>
}

export type BudgetCategory = {
  id: number
  name: string
  subcategories: Array<{ id: number; category_id: number; name: string }>
}

export type BudgetLabelsResponse = { categories: BudgetCategory[] }

export async function fetchBudgetLabels(): Promise<BudgetLabelsResponse> {
  const r = await apiFetch('/api/budget-labels')
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) throw new Error(await readApiErrorMessage(r))
  return r.json() as Promise<BudgetLabelsResponse>
}

export async function createSubcategory(body: {
  category_id: number
  name: string
  match_keywords?: string | null
}): Promise<{ id: number; category_id: number; name: string }> {
  const r = await apiFetch('/api/subcategories', { method: 'POST', body: JSON.stringify(body) })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    throw new Error(await readApiErrorMessage(r))
  }
  return r.json() as Promise<{ id: number; category_id: number; name: string }>
}

export async function deleteSubcategory(id: number): Promise<void> {
  const r = await apiFetch(`/api/subcategories/${id}`, { method: 'DELETE' })
  if (r.status === 401) throw new Error('UNAUTHORIZED')
  if (!r.ok) {
    throw new Error(await readApiErrorMessage(r))
  }
}
