import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  authMe,
  fetchDashboard,
  fetchUncategorized,
  importPdf,
  logout,
  mapTransaction,
  type DashboardResponse,
} from '../api/client'
import ChartErrorBoundary from '../components/ChartErrorBoundary'

const SankeyPlot = lazy(() => import('../components/SankeyPlot'))

const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
]

const LAST_IMPORT_KEY = 'ft_last_import_id'

export default function DashboardPage() {
  const nav = useNavigate()
  const qc = useQueryClient()
  const now = useMemo(() => new Date(), [])
  const [year, setYear] = useState(now.getFullYear())
  const [month, setMonth] = useState(now.getMonth() + 1)
  const [categories, setCategories] = useState<string[]>([])
  const [subcategories, setSubcategories] = useState<string[]>([])
  const [lastImportId, setLastImportId] = useState<number | null>(() => {
    const raw = sessionStorage.getItem(LAST_IMPORT_KEY)
    if (!raw) return null
    const n = parseInt(raw, 10)
    return Number.isFinite(n) ? n : null
  })
  const [mapTxnId, setMapTxnId] = useState<number | null>(null)
  const [mapSubcatId, setMapSubcatId] = useState<number | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  /** Must be true before any API call that needs the session cookie (avoids racing /auth/me). */
  const [authReady, setAuthReady] = useState(false)

  const redirectLoginRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    void authMe().then((ok) => {
      if (cancelled) return
      if (!ok) {
        nav('/login', { replace: true })
        return
      }
      setAuthReady(true)
    })
    return () => {
      cancelled = true
    }
  }, [nav])

  const dashboardQuery = useQuery({
    queryKey: ['dashboard', year, month, categories, subcategories],
    queryFn: () =>
      fetchDashboard({
        year,
        month,
        categories: categories.length ? categories : undefined,
        subcategories: subcategories.length ? subcategories : undefined,
      }),
    enabled: authReady,
    staleTime: 0,
    refetchOnMount: 'always',
    refetchOnWindowFocus: true,
    gcTime: 10 * 60_000,
  })

  /** Never call navigate() during render — it caused blank screens when API returned 401 mid-session. */
  useEffect(() => {
    const err = dashboardQuery.error
    if (!err || !(err instanceof Error) || err.message !== 'UNAUTHORIZED') return
    if (redirectLoginRef.current) return
    redirectLoginRef.current = true
    nav('/login', { replace: true })
  }, [dashboardQuery.error, nav])

  const uncategorizedQuery = useQuery({
    queryKey: ['uncategorized', lastImportId],
    queryFn: () => fetchUncategorized(lastImportId!),
    enabled: authReady && lastImportId != null,
  })

  const importMutation = useMutation({
    mutationFn: (file: File) => importPdf(file),
    onSuccess: (data) => {
      sessionStorage.setItem(LAST_IMPORT_KEY, String(data.import_id))
      setLastImportId(data.import_id)
      void qc.invalidateQueries({ queryKey: ['dashboard'] })
      void qc.invalidateQueries({ queryKey: ['uncategorized', data.import_id] })
    },
  })

  const mapMutation = useMutation({
    mutationFn: ({ tid, sid }: { tid: number; sid: number }) => mapTransaction(tid, sid),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['dashboard'] })
      void qc.invalidateQueries({ queryKey: ['uncategorized', lastImportId] })
      setMapTxnId(null)
      setMapSubcatId(null)
    },
  })

  async function onLogout() {
    await logout()
    nav('/login', { replace: true })
  }

  if (!authReady) {
    return (
      <div className="page-pad">
        <p className="muted">Checking session…</p>
      </div>
    )
  }

  if (dashboardQuery.isLoading) {
    return <div className="page-pad"><p className="muted">Loading dashboard…</p></div>
  }

  if (dashboardQuery.error) {
    const msg =
      dashboardQuery.error instanceof Error ? dashboardQuery.error.message : 'Error'
    if (msg === 'UNAUTHORIZED') {
      return (
        <div className="page-pad">
          <p className="muted">Session expired — returning to sign in…</p>
        </div>
      )
    }
    return (
      <div className="page-pad">
        <p className="error">{msg}</p>
      </div>
    )
  }

  const d = dashboardQuery.data as DashboardResponse

  const yearOptions = [now.getFullYear() - 1, now.getFullYear(), now.getFullYear() + 1]

  return (
    <div className="page-pad dashboard">
      <header className="page-header">
        <div className="header-brand">
          <h1>Finance Tracker</h1>
          <div className="header-countdown-wrap">
            <span className="status-pill">{d.salary_countdown_label}</span>
            <p
              className="header-countdown-sub muted"
              title="KPIs and charts use the global pay window from Settings. The countdown picks the nearest pay among your income streams (each can have its own pay day)."
            >
              Next pay: <strong className="header-countdown-date">{d.due_salary_date}</strong>
              <span className="header-countdown-sep"> · </span>
              <span className="header-countdown-hint">Sums use your global salary window.</span>
            </p>
          </div>
        </div>
        <div className="header-actions">
          <Link to="/settings" className="btn ghost">
            Settings
          </Link>
          <button type="button" className="btn ghost btn-logout" onClick={() => void onLogout()}>
            Log out
          </button>
        </div>
      </header>

      {d.expected_income_net != null && d.expected_income_net > 0 && (
        <div className="expected-income-banner">
          Expected income (configured): <strong>{d.expected_income_net.toLocaleString()} PLN</strong>
          {d.income_variance_vs_expected_percent != null && (
            <>
              {' '}
              · vs actual in window:{' '}
              <strong>
                {d.income_variance_vs_expected_percent > 0 ? '+' : ''}
                {d.income_variance_vs_expected_percent}%
              </strong>
            </>
          )}
        </div>
      )}

      <section className="panel kpi-deck">
        <div className="panel-header kpi-deck-header">
          <div>
            <h2>At a glance</h2>
            <p className="panel-desc">
              Salary-window snapshot for {MONTHS[month - 1]} {year}. Values stay 0 until you import transactions or set
              expected income in Settings.
            </p>
          </div>
        </div>
        <div className="kpi-grid">
          {d.kpis.map((k, i) => (
            <div
              key={k.title}
              data-kpi={k.kind ?? 'default'}
              className={`kpi-card kpi-card--kind-${k.kind ?? 'default'}${i === 0 ? ' kpi-card--hero' : ''}`}
            >
              <div className="kpi-title">{k.title}</div>
              <div className="kpi-value">{k.value}</div>
              <div className="kpi-sub">{k.subtitle}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="dash-toolbar">
        <div className="dash-toolbar-period">
          <label className="label label--inline">
            <span className="label-text">Period</span>
            <select className="input input--compact" value={year} onChange={(e) => setYear(parseInt(e.target.value, 10))}>
              {yearOptions.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </label>
          <label className="label label--inline">
            <select className="input input--compact" value={month} onChange={(e) => setMonth(parseInt(e.target.value, 10))}>
              {MONTHS.map((name, i) => (
                <option key={name} value={i + 1}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <button type="button" className="btn ghost dash-toolbar-filters-btn" onClick={() => setFiltersOpen((o) => !o)}>
          {filtersOpen ? 'Hide' : 'Narrow by'} category / sub-category
          <span className="dash-toolbar-chevron" aria-hidden>
            {filtersOpen ? '▴' : '▾'}
          </span>
        </button>
        {(categories.length > 0 || subcategories.length > 0) && (
          <span className="dash-toolbar-active-filters muted small">
            {categories.length + subcategories.length} filter{categories.length + subcategories.length === 1 ? '' : 's'} active
          </span>
        )}
      </div>

      {filtersOpen ? (
        <section className="panel filters-panel filters-panel--secondary">
          <div className="panel-header">
            <h2>Optional filters</h2>
            <p className="panel-desc">Refine the chart and tables. Leave empty to see everything.</p>
          </div>
          <div className="filter-row-lists">
            <label className="label grow">
              Categories
              <select
                className="input"
                multiple
                size={Math.min(6, Math.max(3, d.filter_categories.length || 3))}
                value={categories}
                onChange={(e) => {
                  const opts = Array.from(e.target.selectedOptions).map((o) => o.value)
                  setCategories(opts)
                  setSubcategories([])
                }}
              >
                {d.filter_categories.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="label grow">
              Sub-categories
              <select
                className="input"
                multiple
                size={Math.min(6, Math.max(3, d.filter_subcategories.length || 3))}
                value={subcategories}
                onChange={(e) =>
                  setSubcategories(Array.from(e.target.selectedOptions).map((o) => o.value))
                }
              >
                {d.filter_subcategories.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="hint">
            <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+click for multiple. Clear selections to reset.
          </p>
        </section>
      ) : null}

      <div className="grid-3">
        <section className="panel">
          <div className="panel-header">
            <h2>Calendar</h2>
            <p className="panel-desc">
              Holidays, today, and pay days (global + each income stream with its own pay day in Settings).
            </p>
          </div>
          <div className="calendar-html" dangerouslySetInnerHTML={{ __html: d.calendar_html }} />
        </section>
        <section className="panel metrics-side">
          <div className="panel-header">
            <h2>Quick figures</h2>
            <p className="panel-desc">Key amounts for this view.</p>
          </div>
          <div className="metric-line">
            <span className="metric-label">Next pay (nearest)</span>
            <span className="metric-val">{d.due_salary_date}</span>
          </div>
          <div className="metric-line">
            <span className="metric-label">Groceries</span>
            <span className="metric-val">{d.groceries_amount.toLocaleString()} PLN</span>
          </div>
          <div className="metric-line">
            <span className="metric-label">Net of net</span>
            <span className="metric-val">{d.net_of_net.toLocaleString()} PLN</span>
          </div>
          <div className="metric-line">
            <span className="metric-label">Need %</span>
            <span className="metric-val">{d.needs_percent.toFixed(2)} %</span>
          </div>
          <div className="metric-line">
            <span className="metric-label">ΔTarget %</span>
            <span className="metric-val">{d.target_vs_actual_percent.toFixed(2)} %</span>
          </div>
          <div className="metric-line">
            <span className="metric-label">Import quality</span>
            <span className="metric-val">
              {d.import_quality_value}
              <span className="muted small"> — {d.import_quality_sub}</span>
            </span>
          </div>
        </section>
        <section className="panel chart-panel">
          <div className="panel-header">
            <h2>Income → expenses</h2>
            <p className="panel-desc">Flow between income categories and spending.</p>
          </div>
          <Suspense fallback={<p className="muted">Loading chart…</p>}>
            <ChartErrorBoundary>
              <SankeyPlot figure={d.sankey_plotly} />
            </ChartErrorBoundary>
          </Suspense>
        </section>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>To pay</h2>
          <p className="panel-desc">Planned vs actual in the current salary window.</p>
        </div>
        {d.to_pay_empty_reason ? (
          <div className="empty-state">
            <strong>Nothing scheduled</strong>
            {d.to_pay_empty_reason}
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Sub-category</th>
                  <th>Deadline</th>
                  <th>Planned</th>
                  <th>Actual</th>
                  <th>Variance</th>
                </tr>
              </thead>
              <tbody>
                {d.to_pay_rows.map((row, i) => (
                  <tr key={`${row.SubCategory}-${i}`}>
                    <td>{row.Category}</td>
                    <td>{row.SubCategory}</td>
                    <td>{row.Deadline ?? '—'}</td>
                    <td>{row.PlannedAmount.toFixed(2)}</td>
                    <td>{row.ActualAmount.toFixed(2)}</td>
                    <td>{row.Variance.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="hint">
              Variance = Actual − Planned (positive means over budget).
            </p>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>PDF import</h2>
          <p className="panel-desc">Upload a bank statement PDF to add transactions.</p>
        </div>
        <div className="file-upload">
          <input
            id="pdf-import"
            className="file-upload-input"
            type="file"
            accept="application/pdf,.pdf"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) importMutation.mutate(f)
              e.target.value = ''
            }}
          />
          <label htmlFor="pdf-import" className="btn primary file-upload-label">
            Choose PDF file
          </label>
          <span className="file-hint">Only .pdf</span>
        </div>
        {importMutation.isPending ? <p className="muted">Importing…</p> : null}
        {importMutation.error ? (
          <p className="error">
            {importMutation.error instanceof Error
              ? importMutation.error.message
              : 'Import failed'}
          </p>
        ) : null}
        {importMutation.isSuccess ? (
          <p className="ok">
            Imported {importMutation.data.rows_added} rows (warnings:{' '}
            {importMutation.data.parse_warnings_count})
          </p>
        ) : null}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Manual category mapping</h2>
          <p className="panel-desc">Assign a subcategory to lines the parser could not classify.</p>
        </div>
        {!lastImportId ? (
          <div className="empty-state">
            <strong>Import a PDF first</strong>
            After a successful import, uncategorized transactions appear here for mapping.
          </div>
        ) : uncategorizedQuery.isLoading ? (
          <p className="muted">Loading uncategorized…</p>
        ) : uncategorizedQuery.error ? (
          <p className="error">Failed to load uncategorized</p>
        ) : uncategorizedQuery.data?.transactions.length === 0 ? (
          <p className="muted">All transactions are categorized for this import.</p>
        ) : (
          <div className="map-grid">
            <label className="label">
              Transaction
              <select
                className="input"
                value={mapTxnId ?? ''}
                onChange={(e) => setMapTxnId(parseInt(e.target.value, 10))}
              >
                <option value="">Select…</option>
                {uncategorizedQuery.data!.transactions.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="label">
              Map to subcategory
              <select
                className="input"
                value={mapSubcatId ?? ''}
                onChange={(e) => setMapSubcatId(parseInt(e.target.value, 10))}
              >
                <option value="">Select…</option>
                {uncategorizedQuery.data!.subcategories.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="btn primary"
              disabled={mapMutation.isPending || mapTxnId == null || mapSubcatId == null}
              onClick={() => {
                if (mapTxnId != null && mapSubcatId != null) {
                  mapMutation.mutate({ tid: mapTxnId, sid: mapSubcatId })
                }
              }}
            >
              Save mapping
            </button>
            {mapMutation.error ? (
              <p className="error">
                {mapMutation.error instanceof Error ? mapMutation.error.message : 'Error'}
              </p>
            ) : null}
            {mapMutation.isSuccess ? <p className="ok">Mapping saved.</p> : null}
          </div>
        )}
      </section>
    </div>
  )
}
