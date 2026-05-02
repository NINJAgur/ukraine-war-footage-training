import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/',            component: () => import('../views/PublicFeed.vue') },
  { path: '/archive',     component: () => import('../views/Archive.vue') },
  { path: '/submit',      component: () => import('../views/Submit.vue') },
  { path: '/admin/login', component: () => import('../views/admin/AdminLogin.vue') },
  { path: '/admin',       component: () => import('../views/admin/AdminPanel.vue'), meta: { requiresAuth: true } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: (to, _from, savedPosition) => {
    if (savedPosition) return savedPosition
    if (to.hash) return { el: to.hash, behavior: 'smooth' }
    return { top: 0 }
  },
})

router.beforeEach((to) => {
  if (to.meta.requiresAuth && !localStorage.getItem('token')) {
    return '/admin/login'
  }
})

export default router
