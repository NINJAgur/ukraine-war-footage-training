import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import TickerBar from '@/components/TickerBar.vue'

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({ models: {} }),
  }))
})

describe('TickerBar', () => {
  it('renders without crash', () => {
    const wrapper = mount(TickerBar)
    expect(wrapper.exists()).toBe(true)
  })

  it('shows LIVE label', () => {
    const wrapper = mount(TickerBar)
    expect(wrapper.text()).toContain('LIVE')
  })

  it('shows base status entries', async () => {
    const wrapper = mount(TickerBar)
    await flushPromises()
    expect(wrapper.text()).toContain('ARCHIVE ONLINE')
  })

  it('shows FUNKER530 entry', async () => {
    const wrapper = mount(TickerBar)
    await flushPromises()
    expect(wrapper.text()).toContain('FUNKER530')
  })

  it('shows detection classes entry', async () => {
    const wrapper = mount(TickerBar)
    await flushPromises()
    expect(wrapper.text()).toContain('AIRCRAFT')
    expect(wrapper.text()).toContain('VEHICLE')
    expect(wrapper.text()).toContain('PERSONNEL')
  })

  it('shows mAP50 when model stats returned', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        models: {
          AIRCRAFT: { status: 'DONE', map50: 0.929 },
        },
      }),
    }))
    const wrapper = mount(TickerBar)
    await flushPromises()
    expect(wrapper.text()).toContain('0.929')
  })

  it('shows QUEUED when model has no stats', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ models: {} }),
    }))
    const wrapper = mount(TickerBar)
    await flushPromises()
    expect(wrapper.text()).toContain('QUEUED')
  })
})
