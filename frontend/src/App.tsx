import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import Home from './pages/Home'

// the analysis page carries the charts and Meta's video player: load it only when it is opened
const Analyze = lazy(() => import('./pages/Analyze'))
const ApiPage = lazy(() => import('./pages/ApiPage'))

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<p className="p-10 text-center text-ink-soft" role="status">Loading…</p>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/analyze/:videoId" element={<Analyze />} />
          <Route path="/developers" element={<ApiPage />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
