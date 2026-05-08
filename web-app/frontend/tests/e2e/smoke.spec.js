import { test, expect } from '@playwright/test'

test('home page loads and has expected content', async ({ page }) => {
  await page.goto('/')
  // Either title contains UKRARCHIVE or navigation landmark is present
  const title = await page.title()
  const hasNav = await page.locator('nav').count()
  const hasH1 = await page.locator('h1').count()
  expect(title.toUpperCase().includes('UKRARCHIVE') || hasNav > 0 || hasH1 > 0).toBe(true)
})

test('archive page loads', async ({ page }) => {
  await page.goto('/archive')
  await page.waitForLoadState('networkidle')
  // Page should render without error — either has content or is loading
  const body = await page.locator('body').textContent()
  expect(body.length).toBeGreaterThan(0)
})

test('submit page has a form', async ({ page }) => {
  await page.goto('/submit')
  await page.waitForLoadState('networkidle')
  const form = await page.locator('form, input[type="text"], input[type="url"], textarea').count()
  expect(form).toBeGreaterThan(0)
})
