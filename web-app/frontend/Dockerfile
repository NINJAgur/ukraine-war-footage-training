# Stage 1: build Vue app
FROM node:20-alpine AS builder

WORKDIR /app
COPY web-app/frontend/package*.json ./
RUN npm ci
COPY web-app/frontend/ ./
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY web-app/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
