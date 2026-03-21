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
    <div className="page-pad dashboard">
      <header className="page-header">
        <div>
          <h1>Settings &amp; onboarding</h1>
          <span className="status-pill">{STRATEGIES.find((s) => s.value === strategy)?.label ?? strategy}</span>
        </div>
        <div className="header-actions">
          <Link to="/" className="btn ghost">
            Dashboard
          </Link>
          <button type="button" className="btn ghost btn-logout" onClick={() => void onLogout()}>
            Log out
          </button>
        </div>
      </header>

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
            {saveSettings.isPending ? 'Saving…' : 'Save pay & budget settings'}
          </button>
          <button
            type="button"
            className="btn primary settings-cta-dash"
            disabled={saveSettings.isPending}
            onClick={() =>
              saveSettings.mutate(undefined, {
                onSuccess: () => {
                  void qc.invalidateQueries({ queryKey: ['dashboard'] })
                  nav('/')
                },
              })
            }
          >
            Save &amp; open dashboard
          </button>
        </div>
        {saveSettings.isError && <p className="error">{(saveSettings.error as Error).message}</p>}
        <p className="hint settings-hint">
          Pay day and strategy apply to the calendar, salary window, and KPIs as soon as you save. Use &quot;Save &amp; open
          dashboard&quot; to return and see them immediately.
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
          {(incomeQ.data ?? []).map((row: IncomeSource) => (
            <IncomeRow key={row.id} row={row} onChanged={() => void qc.invalidateQueries({ queryKey: ['dashboard'] })} />
          ))}
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
              {(recurringQ.data ?? []).map((r: RecurringExpenseRow) => (
                <RecurringEditor key={r.id} row={r} />
              ))}
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
    </div>
  )
}

function IncomeRow({ row, onChanged }: { row: IncomeSource; onChanged: () => void }) {
  const qc = useQueryClient()
  const [net, setNet] = useState(String(row.net_amount ?? ''))
  const [gross, setGross] = useState(String(row.gross_amount ?? ''))
  const [label, setLabel] = useState(row.label)

  const patch = useMutation({
    mutationFn: () =>
      patchIncomeSource(row.id, {
        label,
        net_amount: net ? Number(net) : undefined,
        gross_amount: gross ? Number(gross) : undefined,
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['income-sources'] })
      onChanged()
    },
  })

  const del = useMutation({
    mutationFn: () => deleteIncomeSource(row.id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['income-sources'] })
      onChanged()
    },
  })

  return (
    <li className="settings-list-item">
      <input className="input" value={label} onChange={(e) => setLabel(e.target.value)} />
      <span className="muted">{row.contract_type}</span>
      <input
        className="input"
        type="number"
        placeholder="Gross"
        value={gross}
        onChange={(e) => setGross(e.target.value)}
      />
      <input className="input" type="number" placeholder="Net" value={net} onChange={(e) => setNet(e.target.value)} />
      <button type="button" className="btn ghost" onClick={() => patch.mutate()}>
        Save
      </button>
      <button type="button" className="btn ghost" onClick={() => del.mutate()}>
        Delete
      </button>
    </li>
  )
}

function RecurringEditor({ row }: { row: RecurringExpenseRow }) {
  const qc = useQueryClient()
  const [amt, setAmt] = useState(row.planned_amount ?? '')
  const [day, setDay] = useState(row.planned_deadline_day ?? '')

  const patch = useMutation({
    mutationFn: () =>
      patchRecurringExpense(row.id, {
        planned_amount: amt === '' ? undefined : Number(amt),
        planned_deadline_day: day === '' ? undefined : Number(day),
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
          value={amt}
          onChange={(e) => setAmt(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </td>
      <td>
        <input
          className="input table-input"
          type="number"
          min={1}
          max={31}
          value={day}
          onChange={(e) => setDay(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </td>
      <td>
        <button type="button" className="btn ghost" onClick={() => patch.mutate()}>
          Save
        </button>
      </td>
    </tr>
  )
}
