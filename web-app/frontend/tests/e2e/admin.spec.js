import { test, expect } from '@playwright/test'

test('admin panel accessible after login', async ({ page }) => {
  await page.goto('/admin')
  await page.waitForLoadState('domcontentloaded')

  // If redirected to login page, fill credentials and submit
  const isLoginPage = await page.locator('input[type="password"]').count()
  if (isLoginPage > 0) {
    await page.locator('input[type="text"], input[placeholder*="user" i], input[name="username"]').first().fill('admin')
    await page.locator('input[type="password"]').fill('admin123')
    await page.locator('button[type="submit"], button').first().click()
    await page.waitForLoadState('networkidle')
  }

  // Verify admin panel is shown
  const heading = page.locator('h1, h2, h3').filter({ hasText: /TRAINING CONTROL|ADMIN|PANEL/i })
  await expect(heading.first()).toBeVisible({ timeout: 8000 })
})

test('admin panel shows model cards', async ({ page }) => {
  await page.goto('/admin')
  await page.waitForLoadState('domcontentloaded')

  const isLoginPage = await page.locator('input[type="password"]').count()
  if (isLoginPage > 0) {
    await page.locator('input[type="text"], input[placeholder*="user" i], input[name="username"]').first().fill('admin')
    await page.locator('input[type="password"]').fill('admin123')
    await page.locator('button[type="submit"], button').first().click()
    await page.waitForLoadState('networkidle')
  }

  // Admin panel should have model-related content
  const modelText = page.locator('body').filter({ hasText: /AIRCRAFT|VEHICLE|PERSONNEL/i })
  await expect(modelText).toBeVisible({ timeout: 8000 })
})
