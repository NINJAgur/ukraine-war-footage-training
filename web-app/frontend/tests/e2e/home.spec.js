import { test, expect } from '@playwright/test'

test('home page loads', async ({ page }) => {
  await page.goto('/')
  await page.waitForLoadState('domcontentloaded')
  await expect(page).not.toHaveURL(/error/i)
})

test('home page has nav element', async ({ page }) => {
  await page.goto('/')
  await page.waitForLoadState('domcontentloaded')
  const nav = page.locator('nav')
  await expect(nav).toBeVisible()
})

test('home page has hero section', async ({ page }) => {
  await page.goto('/')
  await page.waitForLoadState('domcontentloaded')
  // Hero section contains h1 or section.hero
  const hero = page.locator('section.hero, h1').first()
  await expect(hero).toBeVisible()
})

test('stats or ML card section loads', async ({ page }) => {
  await page.goto('/')
  // Wait up to 10s for ML card or stats to appear
  const card = page.locator('.ml-card, .hero-stat').first()
  await expect(card).toBeVisible({ timeout: 10000 })
})
