import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import HeroSection from '@/components/HeroSection.vue'

const FAKE_CLIPS = [
  { videoUrl: '/media/annotated/AIRCRAFT/2025-01-01/abc_annotated.mp4', detClass: 'AIRCRAFT' },
  { videoUrl: '/media/annotated/VEHICLE/2025-01-01/def_annotated.mp4',  detClass: 'VEHICLE' },
]

const FAKE_STATS = {
  clips_total: 142,
  images_labeled: 26000,
  models: {
    AIRCRAFT:  { status: 'DONE', map50: 0.929, images: 8000 },
    VEHICLE:   { status: 'DONE', map50: 0.871, images: 6000 },
    PERSONNEL: { status: 'DONE', map50: 0.780, images: 5000 },
    GENERAL:   { status: 'QUEUED', map50: null, images: 0 },
  },
}

function makeFetch(statsOk = true, clipsData = FAKE_CLIPS) {
  return vi.fn().mockImplementation((url) => {
    if (url === '/api/stats') {
      return Promise.resolve({
        ok: statsOk,
        json: () => Promise.resolve(FAKE_STATS),
      })
    }
    if (url === '/api/annotated-clips') {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(clipsData),
      })
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve({}) })
  })
}

describe('HeroSection', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('renders without crash', () => {
    vi.stubGlobal('fetch', makeFetch())
    const wrapper = mount(HeroSection, {
      global: {
        stubs: { 'router-link': true, MLCard: true },
      },
    })
    expect(wrapper.exists()).toBe(true)
  })

  it('sets heroVideoSrc from API response when clips are returned', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const wrapper = mount(HeroSection, {
      global: {
        stubs: { 'router-link': true, MLCard: true },
      },
    })
    await flushPromises()
    // Should pick the first non-GENERAL clip (no GENERAL in FAKE_CLIPS), falling back to clips[0]
    const video = wrapper.find('video')
    expect(video.attributes('src')).toBe(FAKE_CLIPS[0].videoUrl)
  })

  it('falls back to /hero.mp4 when API returns empty array', async () => {
    vi.stubGlobal('fetch', makeFetch(true, []))
    const wrapper = mount(HeroSection, {
      global: {
        stubs: { 'router-link': true, MLCard: true },
      },
    })
    await flushPromises()
    const video = wrapper.find('video')
    expect(video.attributes('src')).toBe('/hero.mp4')
  })

  it('falls back to /hero.mp4 when API is not ok', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, json: () => Promise.resolve([]) }))
    const wrapper = mount(HeroSection, {
      global: {
        stubs: { 'router-link': true, MLCard: true },
      },
    })
    await flushPromises()
    const video = wrapper.find('video')
    expect(video.attributes('src')).toBe('/hero.mp4')
  })

  it('prefers GENERAL clip for heroVideoSrc when available', async () => {
    const clipsWithGeneral = [
      { videoUrl: '/media/annotated/AIRCRAFT/2025-01-01/abc.mp4', detClass: 'AIRCRAFT' },
      { videoUrl: '/media/annotated/GENERAL/2025-01-01/ggg.mp4',  detClass: 'GENERAL' },
    ]
    vi.stubGlobal('fetch', makeFetch(true, clipsWithGeneral))
    const wrapper = mount(HeroSection, {
      global: {
        stubs: { 'router-link': true, MLCard: true },
      },
    })
    await flushPromises()
    const video = wrapper.find('video')
    expect(video.attributes('src')).toBe('/media/annotated/GENERAL/2025-01-01/ggg.mp4')
  })
})
