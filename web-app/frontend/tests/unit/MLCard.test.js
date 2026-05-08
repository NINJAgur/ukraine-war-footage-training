import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import MLCard from '@/components/MLCard.vue'

const MOCK_CAT = {
  id: 'aircraft',
  title: 'Aircraft',
  label: 'Drone, Missile and Aircraft detection',
  desc: 'nc=1 specialist for fixed-wing, rotary-wing, UAV/drone.',
  detections: [],
  videoSrc: null,
  modelInfo: 'YOLOv8m',
}

// Stub IntersectionObserver — not available in jsdom
beforeEach(() => {
  global.IntersectionObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    disconnect: vi.fn(),
    unobserve: vi.fn(),
  }))
})

describe('MLCard', () => {
  it('renders without crash', () => {
    const wrapper = mount(MLCard, {
      props: { cat: MOCK_CAT },
      global: { stubs: { canvas: true } },
    })
    expect(wrapper.exists()).toBe(true)
  })

  it('shows category label text', () => {
    const wrapper = mount(MLCard, {
      props: { cat: MOCK_CAT },
      global: { stubs: { canvas: true } },
    })
    expect(wrapper.text()).toContain('AIRCRAFT')
  })

  it('shows mAP50 value when stats prop provided', () => {
    const wrapper = mount(MLCard, {
      props: {
        cat: MOCK_CAT,
        stats: { map50: 0.929, images: 8000, status: 'DONE' },
      },
      global: { stubs: { canvas: true } },
    })
    expect(wrapper.text()).toContain('0.929')
  })

  it('shows TRAINING status label when status is TRAINING', () => {
    const wrapper = mount(MLCard, {
      props: {
        cat: MOCK_CAT,
        stats: { map50: null, images: 5000, status: 'TRAINING' },
      },
      global: { stubs: { canvas: true } },
    })
    expect(wrapper.text()).toContain('TRAINING')
  })

  it('falls back to modelInfo when no stats prop', () => {
    const wrapper = mount(MLCard, {
      props: { cat: MOCK_CAT, stats: null },
      global: { stubs: { canvas: true } },
    })
    expect(wrapper.text()).toContain('YOLOv8m')
  })

  it('shows category title', () => {
    const wrapper = mount(MLCard, {
      props: { cat: MOCK_CAT },
      global: { stubs: { canvas: true } },
    })
    expect(wrapper.text()).toContain('Aircraft')
  })
})
