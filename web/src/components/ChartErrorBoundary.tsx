import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = { children: ReactNode; fallback?: ReactNode }

type State = { hasError: boolean }

export default class ChartErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Chart render failed:', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="chart-fallback">
            Income → expenses chart could not be rendered (browser or data). Try another month or reload the page.
          </div>
        )
      )
    }
    return this.props.children
  }
}
