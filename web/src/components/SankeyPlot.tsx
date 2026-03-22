import Plot from 'react-plotly.js'
import type { Data, Layout } from 'plotly.js'

type Props = {
  figure: { data?: unknown[]; layout?: Record<string, unknown> }
}

export default function SankeyPlot({ figure }: Props) {
  const data = (figure.data ?? []) as Data[]
  const layout: Partial<Layout> = {
    ...(figure.layout as Partial<Layout> | undefined),
    autosize: true,
    margin: { l: 0, r: 0, t: 8, b: 8 },
  }
  return (
    <Plot
      data={data}
      layout={layout}
      style={{ width: '100%', height: 420 }}
      useResizeHandler
      config={{
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        scrollZoom: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      }}
    />
  )
}
