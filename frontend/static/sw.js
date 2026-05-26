// BurnoutGuard Service Worker — PWA offline support
const CACHE_NAME = 'burnoutguard-v1';
const STATIC_ASSETS = [
  '/',
  '/survey',
  '/dashboard',
  '/static/manifest/manifest.json',
];

// Install: cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS).catch(() => {});
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: network-first, fallback to cache
self.addEventListener('fetch', event => {
  // Only handle GET requests
  if (event.request.method !== 'GET') return;
  // Skip non-http requests
  if (!event.request.url.startsWith('http')) return;

  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Cache successful responses for static assets
        if (response.ok && event.request.url.includes('/static/')) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => {
        // Return cached version if network fails
        return caches.match(event.request).then(cached => {
          if (cached) return cached;
          // Return offline page for navigation requests
          if (event.request.mode === 'navigate') {
            return new Response(
              `<!DOCTYPE html><html><head><meta charset="utf-8">
              <meta name="viewport" content="width=device-width,initial-scale=1">
              <title>BurnoutGuard — Offline</title>
              <style>body{background:#080812;color:#e8e8f0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;text-align:center;}
              h1{font-size:1.5rem;}p{color:#8888aa;}</style></head>
              <body><div><div style="font-size:3rem;margin-bottom:1rem;">📡</div>
              <h1>You're Offline</h1>
              <p>Please connect to the internet to use BurnoutGuard.</p>
              <p style="margin-top:1rem;"><a href="/" style="color:#6c63ff;">Try Again →</a></p>
              </div></body></html>`,
              { headers: { 'Content-Type': 'text/html' } }
            );
          }
        });
      })
  );
});
