import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import Home from '../src/pages/Home'
import ApiPage from '../src/pages/ApiPage'

describe('Home', () => {
  it('states what Gauge is and offers the two ways in', () => {
    render(<MemoryRouter><Home /></MemoryRouter>)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(/measuring instrument/i)
    expect(screen.getByRole('link', { name: 'Try it on a video' })).toHaveAttribute('href', '/analyze')
    expect(screen.getByRole('link', { name: 'Use the API' })).toHaveAttribute('href', '/developers')
  })
  it('is honest that the animation is an illustration', () => {
    render(<MemoryRouter><Home /></MemoryRouter>)
    expect(screen.getByText(/Illustration: a simulated pendulum/)).toBeInTheDocument()
  })
  it('explains that numbers come from code', () => {
    render(<MemoryRouter><Home /></MemoryRouter>)
    expect(screen.getByText(/A language model never calculates/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Measured, not guessed' })).toBeInTheDocument()
  })
})

describe('API page', () => {
  it('shows working examples and every call', () => {
    render(<ApiPage />)
    expect(screen.getByText(/curl -F "file=@my_video.mp4"/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '/docs' })).toHaveAttribute('href', '/docs')
    for (const path of ['/api/videos', '/api/jobs/{id}/events', '/api/jobs/{id}/answers']) expect(screen.getAllByText(path).length).toBeGreaterThan(0)
  })
  it('copies an example to the clipboard', async () => {
    const write = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true })
    render(<ApiPage />)
    ;(await screen.findAllByRole('button', { name: /^Copy/ }))[0].click()
    expect(write).toHaveBeenCalled()
  })
})
