import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  authMe,
  fetchDashboard,
  fetchUncategorized,
  importPdf,
  logout,
  mapTransaction,
  type DashboardResponse,
} from '../api/client'
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
    return raw ? parseInt(raw, 10) : null
  })
  const [mapTxnId, setMapTxnId] = useState<number | null>(null)
  const [mapSubcatId, setMapSubcatId] = useState<number | null>(null)
  /** Must be true before any API call that needs the session cookie (avoids racing /auth/me). */
  const [authReady, setAuthReady] = useState(false)

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
  })

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
      nav('/login', { replace: true })
      return null
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
    <div className="page-pad">
      <header className="top-bar">
        <div>
          <h1>Finance Tracker</h1>
          <p className="muted">{d.days_left_for_infy_label}</p>
        </div>
        <button type="button" className="btn ghost" onClick={() => void onLogout()}>
          Log out
        </button>
      </header>

      <section className="panel filters">
        <div className="field-row">
          <label className="label">
            Year
            <select
              className="input"
              value={year}
              onChange={(e) => setYear(parseInt(e.target.value, 10))}
            >
              {yearOptions.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </label>
          <label className="label">
            Month
            <select
              className="input"
              value={month}
              onChange={(e) => setMonth(parseInt(e.target.value, 10))}
            >
              {MONTHS.map((name, i) => (
                <option key={name} value={i + 1}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="field-row">
          <label className="label grow">
            Categories (optional)
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
            Sub-categories (optional)
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
        <p className="hint">Hold Ctrl (Cmd on Mac) to select multiple categories or sub-categories.</p>
      </section>

      <section className="panel">
        <h2>Important metrics</h2>
        <div className="kpi-grid">
          {d.kpis.map((k) => (
            <div key={k.title} className="kpi-card">
              <div className="kpi-title">{k.title}</div>
              <div className="kpi-value">{k.value}</div>
              <div className="kpi-sub muted">{k.subtitle}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="grid-3">
        <section className="panel">
          <h3>Calendar</h3>
          <div className="calendar-html" dangerouslySetInnerHTML={{ __html: d.calendar_html }} />
        </section>
        <section className="panel metrics-side">
          <div className="metric-line">
            <span className="metric-label">Due salary</span>
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
        <section className="panel">
          <h3>Income → expenses</h3>
          <Suspense fallback={<p className="muted">Loading chart…</p>}>
            <SankeyPlot figure={d.sankey_plotly} />
          </Suspense>
        </section>
      </div>

      <section className="panel">
        <h2>To pay</h2>
        {d.to_pay_empty_reason ? (
          <p className="muted">{d.to_pay_empty_reason}</p>
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
        <h2>PDF import</h2>
        <input
          type="file"
          accept="application/pdf,.pdf"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) importMutation.mutate(f)
            e.target.value = ''
          }}
        />
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
        <h2>Manual category mapping (uncategorized)</h2>
        {!lastImportId ? (
          <p className="muted">Import a PDF to map uncategorized transactions.</p>
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
