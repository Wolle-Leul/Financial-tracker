import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  authMe,
  calculateNet,
  createIncomeSource,
  createSubcategory,
  deleteIncomeSource,
  deleteSubcategory,
  fetchBudgetLabels,
  fetchIncomeSources,
  fetchRecurringExpenses,
  fetchSalaryRuleSettings,
  fetchTrends,
  logout,
  patchIncomeSource,
  patchRecurringExpense,
  patchSalaryRuleSettings,
  postSettingsSync,
  type IncomeSource,
  type RecurringExpenseRow,
} from '../api/client'

const STRATEGIES = [
  { value: 'custom_target_ratio', label: 'Custom target ratio (uses target % below)' },
  { value: 'classic_50_30_20', label: '50/30/20 style (needs ~50%)' },
  { value: 'zero_based', label: 'Zero-based (tight to your stored ratio)' },
  { value: 'salary_window_only', label: 'Salary window focus (your target ratio)' },
]

const CONTRACTS = [
  { value: 'employment_pl', label: 'UoP / employment (PL)' },
  { value: 'b2b_pl', label: 'B2B / self-employed (PL)' },
  { value: 'other', label: 'Other' },
]

export default function SettingsPage() {
  const nav = useNavigate()
  const qc = useQueryClient()
  const [authReady, setAuthReady] = useState(false)

  useEffect(() => {
    let c = false
    void authMe().then((ok) => {
      if (c) return
      if (!ok) {
        nav('/login', { replace: true })
        return
      }
      setAuthReady(true)
    })
    return () => {
      c = true
    }
  }, [nav])

  const settingsQ = useQuery({
    queryKey: ['settings', 'salary-rule'],
    queryFn: fetchSalaryRuleSettings,
    enabled: authReady,
  })

  const incomeQ = useQuery({
    queryKey: ['income-sources'],
    queryFn: fetchIncomeSources,
    enabled: authReady,
  })

  const recurringQ = useQuery({
    queryKey: ['recurring-expenses'],
    queryFn: fetchRecurringExpenses,
    enabled: authReady,
  })

  const trendsQ = useQuery({
    queryKey: ['settings', 'trends'],
    queryFn: () => fetchTrends(12),
    enabled: authReady,
  })

  const labelsQ = useQuery({
    queryKey: ['budget-labels'],
    queryFn: fetchBudgetLabels,
    enabled: authReady,
  })

  const [newLineCat, setNewLineCat] = useState<number | ''>('')
  const [newLineName, setNewLineName] = useState('')
  const [newLineKw, setNewLineKw] = useState('')

  /** Drafts mirror DB rows; global sync sends all of this in one POST /api/settings/sync transaction. */
  const [incomeDrafts, setIncomeDrafts] = useState<
    Record<number, { label: string; net: string; gross: string }>
  >({})
  const [recurringDrafts, setRecurringDrafts] = useState<
    Record<number, { amt: string | number; day: string | number }>
  >({})
  const [syncing, setSyncing] = useState(false)
  const [syncErr, setSyncErr] = useState<string | null>(null)

  useEffect(() => {
    if (!incomeQ.data) return
    setIncomeDrafts((prev) => {
      const next = { ...prev }
      for (const row of incomeQ.data) {
        if (next[row.id] == null) {
          next[row.id] = {
            label: row.label,
            net: String(row.net_amount ?? ''),
            gross: String(row.gross_amount ?? ''),
          }
        }
      }
      return next
    })
  }, [incomeQ.data])

  useEffect(() => {
    if (!recurringQ.data) return
    setRecurringDrafts((prev) => {
      const next = { ...prev }
      for (const row of recurringQ.data) {
        if (next[row.id] == null) {
          next[row.id] = {
            amt: row.planned_amount ?? '',
            day: row.planned_deadline_day ?? '',
          }
        }
      }
      return next
    })
  }, [recurringQ.data])

  async function saveAllToDatabaseAndOpenDashboard() {
    setSyncErr(null)
    setSyncing(true)
    try {
      const income_rows = (incomeQ.data ?? []).map((row) => {
        const d = incomeDrafts[row.id] ?? {
          label: row.label,
          net: String(row.net_amount ?? ''),
          gross: String(row.gross_amount ?? ''),
        }
        return {
          id: row.id,
          label: (d.label.trim() || row.label).slice(0, 120),
          net_amount: d.net === '' ? null : Number(d.net),
          gross_amount: d.gross === '' ? null : Number(d.gross),
        }
      })
      const recurring_rows = (recurringQ.data ?? []).map((row) => {
        const d = recurringDrafts[row.id] ?? {
          amt: row.planned_amount ?? '',
          day: row.planned_deadline_day ?? '',
        }
        return {
          subcategory_id: row.id,
          planned_amount: d.amt === '' ? null : Number(d.amt),
          planned_deadline_day: d.day === '' ? null : Number(d.day),
        }
      })
      await postSettingsSync({
        salary_day_of_month: salaryDay,
        target_ratio: targetRatio,
        budget_strategy: strategy,
        income_rows,
        recurring_rows,
      })
      void qc.invalidateQueries({ queryKey: ['settings', 'salary-rule'] })
      void qc.invalidateQueries({ queryKey: ['income-sources'] })
      void qc.invalidateQueries({ queryKey: ['recurring-expenses'] })
      void qc.invalidateQueries({ queryKey: ['budget-labels'] })
      await qc.refetchQueries({ queryKey: ['dashboard'] })
      nav('/')
    } catch (e) {
      setSyncErr(e instanceof Error ? e.message : 'Sync failed')
    } finally {
      setSyncing(false)
    }
  }

  const addLine = useMutation({
    mutationFn: () =>
      createSubcategory({
        category_id: Number(newLineCat),
        name: newLineName.trim(),
        match_keywords: newLineKw.trim() || undefined,
      }),
    onSuccess: () => {
      setNewLineName('')
      setNewLineKw('')
      void qc.invalidateQueries({ queryKey: ['budget-labels'] })
      void qc.invalidateQueries({ queryKey: ['recurring-expenses'] })
      void qc.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })

  const removeLine = useMutation({
    mutationFn: (id: number) => deleteSubcategory(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['budget-labels'] })
      void qc.invalidateQueries({ queryKey: ['recurring-expenses'] })
      void qc.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })

  const [salaryDay, setSalaryDay] = useState(10)
  const [targetRatio, setTargetRatio] = useState(0.45)
  const [strategy, setStrategy] = useState('custom_target_ratio')

  useEffect(() => {
    if (!settingsQ.data) return
    setSalaryDay(settingsQ.data.salary_day_of_month)
    setTargetRatio(settingsQ.data.target_ratio)
    setStrategy(settingsQ.data.budget_strategy)
  }, [settingsQ.data])

  const saveSettings = useMutation({
    mutationFn: () =>
      patchSalaryRuleSettings({
        salary_day_of_month: salaryDay,
        target_ratio: targetRatio,
        budget_strategy: strategy,
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['settings', 'salary-rule'] })
      void qc.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })

  const [newIncome, setNewIncome] = useState({
    label: 'Salary',
    employer_name: '',
    contract_type: 'employment_pl',
    gross_amount: '' as string | number,
    net_amount: '' as string | number,
    use_net_only: true,
  })

  const addIncome = useMutation({
    mutationFn: () =>
      createIncomeSource({
        label: newIncome.label,
        employer_name: newIncome.employer_name || undefined,
        contract_type: newIncome.contract_type,
        gross_amount: newIncome.use_net_only ? undefined : Number(newIncome.gross_amount) || undefined,
        net_amount: Number(newIncome.net_amount) || undefined,
        use_net_only: newIncome.use_net_only,
        sort_order: 0,
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['income-sources'] })
      void qc.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })

  async function onLogout() {
    await logout()
    nav('/login', { replace: true })
  }

  if (!authReady)
    return (
      <div className="page-pad">
        <p className="muted">Checking session…</p>
      </div>
    )

  if (settingsQ.isLoading || incomeQ.isLoading) {
    return (
      <div className="page-pad">
        <p className="muted">Loading settings…</p>
      </div>
    )
  }

  return (
    <div className="page-pad dashboard settings-page">
      <header className="page-header">
        <div>
          <h1>Settings &amp; onboarding</h1>
          <span className="status-pill">{STRATEGIES.find((s) => s.value === strategy)?.label ?? strategy}</span>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn primary settings-cta-dash"
            disabled={syncing}
            onClick={() => void saveAllToDatabaseAndOpenDashboard()}
          >
            {syncing ? 'Saving to database…' : 'Save all to database → dashboard'}
          </button>
          <Link to="/" className="btn ghost">
            Dashboard
          </Link>
          <button type="button" className="btn ghost btn-logout" onClick={() => void onLogout()}>
            Log out
          </button>
        </div>
      </header>

      <div className="settings-db-banner">
        <strong>Stored in your database (Supabase / Postgres)</strong>
        <p>
          The green <strong>Save all to database → dashboard</strong> button calls <code>POST /api/settings/sync</code>: one
          transaction that writes <strong>salary_rules</strong> (pay day, target %, strategy), every{' '}
          <strong>income_sources</strong> row, and every recurring <strong>subcategories</strong> plan. The dashboard reads
          those tables on load for the salary window, expected-income banner, KPI copy, and &quot;to pay&quot; math. Row-level
          &quot;Save row&quot; still works for quick single-line updates.
        </p>
      </div>
      {syncErr ? <p className="error page-sync-err">{syncErr}</p> : null}

      <section className="panel">
        <div className="panel-header">
          <h2>Pay schedule &amp; budget</h2>
          <p className="panel-desc">Day of month you get paid (weekends/holidays shift like the calendar). Target ratio drives KPI targets.</p>
        </div>
        <div className="filter-row-dates">
          <label className="label">
            Pay day (month)
            <input
              className="input"
              type="number"
              min={1}
              max={31}
              value={salaryDay}
              onChange={(e) => setSalaryDay(parseInt(e.target.value, 10) || 1)}
            />
          </label>
          <label className="label">
            Target ratio (0–1)
            <input
              className="input"
              type="number"
              step="0.05"
              min={0.05}
              max={0.99}
              value={targetRatio}
              onChange={(e) => setTargetRatio(parseFloat(e.target.value) || 0.45)}
            />
          </label>
          <label className="label">
            Budget strategy
            <select className="input" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              {STRATEGIES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="settings-actions-row">
          <button type="button" className="btn primary" disabled={saveSettings.isPending} onClick={() => saveSettings.mutate()}>
            {saveSettings.isPending ? 'Saving…' : 'Save pay fields only'}
          </button>
          <button
            type="button"
            className="btn primary settings-cta-dash"
            disabled={syncing}
            onClick={() => void saveAllToDatabaseAndOpenDashboard()}
          >
            {syncing ? 'Saving…' : 'Save all to database → dashboard'}
          </button>
        </div>
        {saveSettings.isError && <p className="error">{(saveSettings.error as Error).message}</p>}
        <p className="hint settings-hint">
          &quot;Save pay fields only&quot; updates <code>salary_rules</code> via PATCH. Use the green button (here or in the
          header) to sync pay + income + recurring in one go.
        </p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Income sources</h2>
          <p className="panel-desc">Expected monthly net (used vs actual bank income on the dashboard). Use gross + calculator or enter net directly.</p>
        </div>
        <div className="settings-grid">
          <label className="label">
            Label
            <input className="input" value={newIncome.label} onChange={(e) => setNewIncome((p) => ({ ...p, label: e.target.value }))} />
          </label>
          <label className="label">
            Employer (optional)
            <input
              className="input"
              value={newIncome.employer_name}
              onChange={(e) => setNewIncome((p) => ({ ...p, employer_name: e.target.value }))}
            />
          </label>
          <label className="label">
            Contract type
            <select
              className="input"
              value={newIncome.contract_type}
              onChange={(e) => setNewIncome((p) => ({ ...p, contract_type: e.target.value }))}
            >
              {CONTRACTS.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          <label className="label">
            <input
              type="checkbox"
              checked={newIncome.use_net_only}
              onChange={(e) => setNewIncome((p) => ({ ...p, use_net_only: e.target.checked }))}
            />{' '}
            I know my net (skip gross)
          </label>
          {!newIncome.use_net_only && (
            <label className="label">
              Gross (PLN)
              <input
                className="input"
                type="number"
                value={newIncome.gross_amount}
                onChange={(e) => setNewIncome((p) => ({ ...p, gross_amount: e.target.value }))}
              />
            </label>
          )}
          <label className="label">
            Net (PLN)
            <input
              className="input"
              type="number"
              value={newIncome.net_amount}
              onChange={(e) => setNewIncome((p) => ({ ...p, net_amount: e.target.value }))}
            />
          </label>
        </div>
        <div className="filter-row-dates">
          {!newIncome.use_net_only && (
            <button
              type="button"
              className="btn ghost"
              onClick={() => {
                const g = Number(newIncome.gross_amount)
                if (!g) return
                void calculateNet(g, newIncome.contract_type).then((r) => {
                  setNewIncome((p) => ({ ...p, net_amount: r.net }))
                })
              }}
            >
              Estimate net from gross
            </button>
          )}
          <button
            type="button"
            className="btn primary"
            disabled={addIncome.isPending}
            onClick={() => addIncome.mutate()}
          >
            Add income source
          </button>
        </div>
        <ul className="settings-list">
          {(incomeQ.data ?? []).map((row: IncomeSource) => {
            const draft =
              incomeDrafts[row.id] ?? {
                label: row.label,
                net: String(row.net_amount ?? ''),
                gross: String(row.gross_amount ?? ''),
              }
            return (
              <IncomeRow
                key={row.id}
                row={row}
                draft={draft}
                onDraftChange={(d) => setIncomeDrafts((p) => ({ ...p, [row.id]: d }))}
                onChanged={() => void qc.invalidateQueries({ queryKey: ['dashboard'] })}
                onRowDeleted={() => {
                  setIncomeDrafts((p) => {
                    const n = { ...p }
                    delete n[row.id]
                    return n
                  })
                }}
              />
            )
          })}
        </ul>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Budget lines (subcategories)</h2>
          <p className="panel-desc">
            Add or remove named lines (e.g. Netflix, rent). They appear in filters, recurring bills, and PDF keyword mapping.
          </p>
        </div>
        {labelsQ.isLoading && <p className="muted">Loading…</p>}
        {labelsQ.data && labelsQ.data.categories.length === 0 && (
          <p className="muted">No categories yet — seed data should create defaults on first run.</p>
        )}
        {labelsQ.data && labelsQ.data.categories.length > 0 && (
          <>
            <div className="settings-grid budget-line-form">
              <label className="label">
                Category
                <select
                  className="input"
                  value={newLineCat === '' ? '' : newLineCat}
                  onChange={(e) => setNewLineCat(e.target.value === '' ? '' : parseInt(e.target.value, 10))}
                >
                  <option value="">Select…</option>
                  {labelsQ.data.categories.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="label">
                Line name
                <input
                  className="input"
                  placeholder="e.g. Netflix"
                  value={newLineName}
                  onChange={(e) => setNewLineName(e.target.value)}
                />
              </label>
              <label className="label">
                Keywords (optional)
                <input
                  className="input"
                  placeholder="netflix,streaming"
                  value={newLineKw}
                  onChange={(e) => setNewLineKw(e.target.value)}
                />
              </label>
            </div>
            <button
              type="button"
              className="btn primary"
              disabled={addLine.isPending || newLineCat === '' || !newLineName.trim()}
              onClick={() => addLine.mutate()}
            >
              Add budget line
            </button>
            {addLine.isError && <p className="error">{(addLine.error as Error).message}</p>}
            <div className="table-wrap budget-lines-table">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Line</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {labelsQ.data.categories.flatMap((c) =>
                    c.subcategories.map((s) => (
                      <tr key={s.id}>
                        <td>{c.name}</td>
                        <td>{s.name}</td>
                        <td>
                          <button
                            type="button"
                            className="btn ghost"
                            disabled={removeLine.isPending}
                            onClick={() => {
                              if (window.confirm(`Remove budget line "${s.name}"?`)) removeLine.mutate(s.id)
                            }}
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Recurring bills</h2>
          <p className="panel-desc">Planned amounts and deadline day-of-month (same as budget categories).</p>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Subcategory</th>
                <th>Planned PLN</th>
                <th>Day of month</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(recurringQ.data ?? []).map((r: RecurringExpenseRow) => {
                const draft =
                  recurringDrafts[r.id] ?? {
                    amt: r.planned_amount ?? '',
                    day: r.planned_deadline_day ?? '',
                  }
                return (
                  <RecurringEditor
                    key={r.id}
                    row={r}
                    draft={draft}
                    onDraftChange={(d) => setRecurringDrafts((p) => ({ ...p, [r.id]: d }))}
                  />
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Historical trends (bank data)</h2>
          <p className="panel-desc">Monthly income vs expenses from imported transactions.</p>
        </div>
        {trendsQ.isLoading && <p className="muted">Loading trends…</p>}
        {trendsQ.data && trendsQ.data.months.length === 0 && <p className="muted">No transactions yet — import a PDF.</p>}
        {trendsQ.data && trendsQ.data.months.length > 0 && (
          <div className="trend-chart">
            {trendsQ.data.months.map((m) => {
              const maxVal = Math.max(m.income, m.expenses, 1)
              return (
                <div key={m.month} className="trend-month">
                  <div className="trend-label">{m.month}</div>
                  <div className="trend-bars">
                    <div
                      className="trend-bar income"
                      style={{ height: `${(m.income / maxVal) * 100}%` }}
                      title={`Income ${m.income.toFixed(0)}`}
                    />
                    <div
                      className="trend-bar expense"
                      style={{ height: `${(m.expenses / maxVal) * 100}%` }}
                      title={`Expenses ${m.expenses.toFixed(0)}`}
                    />
                  </div>
                  <div className="trend-net">{m.net >= 0 ? '+' : ''}{m.net.toFixed(0)}</div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      <div className="panel settings-footer-note">
        <p className="muted small">
          Same as the header: use <strong>Save all to database → dashboard</strong> after editing any section so pay
          schedule, income, and recurring plans all reach the database before the dashboard reloads.
        </p>
      </div>
    </div>
  )
}

function IncomeRow({
  row,
  draft,
  onDraftChange,
  onChanged,
  onRowDeleted,
}: {
  row: IncomeSource
  draft: { label: string; net: string; gross: string }
  onDraftChange: (d: { label: string; net: string; gross: string }) => void
  onChanged: () => void
  onRowDeleted: () => void
}) {
  const qc = useQueryClient()

  const patch = useMutation({
    mutationFn: () =>
      patchIncomeSource(row.id, {
        label: draft.label,
        net_amount: draft.net ? Number(draft.net) : undefined,
        gross_amount: draft.gross ? Number(draft.gross) : undefined,
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['income-sources'] })
      onChanged()
    },
  })

  const del = useMutation({
    mutationFn: () => deleteIncomeSource(row.id),
    onSuccess: () => {
      onRowDeleted()
      void qc.invalidateQueries({ queryKey: ['income-sources'] })
      onChanged()
    },
  })

  return (
    <li className="settings-list-item">
      <input
        className="input"
        value={draft.label}
        onChange={(e) => onDraftChange({ ...draft, label: e.target.value })}
      />
      <span className="muted">{row.contract_type}</span>
      <input
        className="input"
        type="number"
        placeholder="Gross"
        value={draft.gross}
        onChange={(e) => onDraftChange({ ...draft, gross: e.target.value })}
      />
      <input
        className="input"
        type="number"
        placeholder="Net"
        value={draft.net}
        onChange={(e) => onDraftChange({ ...draft, net: e.target.value })}
      />
      <button type="button" className="btn ghost" onClick={() => patch.mutate()}>
        Save row
      </button>
      <button type="button" className="btn ghost" onClick={() => del.mutate()}>
        Delete
      </button>
    </li>
  )
}

function RecurringEditor({
  row,
  draft,
  onDraftChange,
}: {
  row: RecurringExpenseRow
  draft: { amt: string | number; day: string | number }
  onDraftChange: (d: { amt: string | number; day: string | number }) => void
}) {
  const qc = useQueryClient()

  const patch = useMutation({
    mutationFn: () =>
      patchRecurringExpense(row.id, {
        planned_amount: draft.amt === '' ? undefined : Number(draft.amt),
        planned_deadline_day: draft.day === '' ? undefined : Number(draft.day),
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['recurring-expenses'] })
      void qc.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })

  return (
    <tr>
      <td>{row.category_name}</td>
      <td>{row.name}</td>
      <td>
        <input
          className="input table-input"
          type="number"
          value={draft.amt}
          onChange={(e) =>
            onDraftChange({ ...draft, amt: e.target.value === '' ? '' : Number(e.target.value) })
          }
        />
      </td>
      <td>
        <input
          className="input table-input"
          type="number"
          min={1}
          max={31}
          value={draft.day}
          onChange={(e) =>
            onDraftChange({ ...draft, day: e.target.value === '' ? '' : Number(e.target.value) })
          }
        />
      </td>
      <td>
        <button type="button" className="btn ghost" onClick={() => patch.mutate()}>
          Save row
        </button>
      </td>
    </tr>
  )
}
