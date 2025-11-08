# Guía de Troubleshooting - Problemas Comunes

## Problemas de Performance

### PROBLEMA: Alta latencia en respuestas de API

**Síntomas:**
- Tiempos de respuesta > 2 segundos
- Timeouts frecuentes
- Quejas de usuarios sobre lentitud

**Diagnóstico:**
1. Verificar métricas de APM (Application Performance Monitoring)
2. Analizar logs de aplicación y base de datos
3. Revisar utilización de CPU y memoria
4. Comprobar queries de base de datos lentas

**Soluciones:**

**Solución 1: Optimización de Queries**
```sql
-- Antes (query lenta)
SELECT * FROM usuarios u 
JOIN pedidos p ON u.id = p.usuario_id 
WHERE u.activo = 1;

-- Después (query optimizada con índices)
SELECT u.id, u.nombre, p.total 
FROM usuarios u 
JOIN pedidos p ON u.id = p.usuario_id 
WHERE u.activo = 1 
AND u.fecha_creacion > '2024-01-01';

-- Agregar índice
CREATE INDEX idx_usuarios_activo_fecha ON usuarios(activo, fecha_creacion);
```

**Solución 2: Implementar Caché**
```python
import redis
from functools import wraps

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Intentar obtener del caché
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Ejecutar función y guardar en caché
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

**Solución 3: Conexión Pool de Base de Datos**
```python
# Configuración optimizada de conexiones
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}
```

### PROBLEMA: Memory Leaks en aplicación Node.js

**Síntomas:**
- Uso de memoria crece constantemente
- Performance degradada en el tiempo
- Eventual crash por falta de memoria

**Diagnóstico:**
```bash
# Monitorear uso de memoria
node --inspect --max-old-space-size=4096 app.js

# Generar heap dump
kill -USR2 <node_process_id>

# Analizar con herramientas
node --prof app.js
node --prof-process isolate-0x[...].log > profile.txt
```

**Soluciones:**

**Solución 1: Limpiar Event Listeners**
```javascript
// Problemático
class MyComponent {
    constructor() {
        window.addEventListener('resize', this.handleResize);
    }
}

// Correcto
class MyComponent {
    constructor() {
        this.handleResize = this.handleResize.bind(this);
        window.addEventListener('resize', this.handleResize);
    }
    
    destroy() {
        window.removeEventListener('resize', this.handleResize);
    }
}
```

**Solución 2: Gestión de Streams**
```javascript
// Problemático
const fs = require('fs');
const stream = fs.createReadStream('large-file.txt');
// Stream nunca se cierra

// Correcto
const fs = require('fs');
const stream = fs.createReadStream('large-file.txt');
stream.on('end', () => stream.close());
stream.on('error', () => stream.close());
```

## Problemas de Conectividad

### PROBLEMA: Errores de conexión a base de datos

**Síntomas:**
- Error: "Connection refused"
- Error: "Too many connections"
- Timeouts intermitentes

**Diagnóstico:**
```bash
# Verificar conectividad
telnet db-server 5432

# Verificar conexiones activas
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Revisar logs de PostgreSQL
tail -f /var/log/postgresql/postgresql.log
```

**Soluciones:**

**Solución 1: Configurar Connection Pooling**
```python
import psycopg2.pool

# Crear pool de conexiones
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host="localhost",
    database="mydb",
    user="user",
    password="password"
)

def get_db_connection():
    return connection_pool.getconn()

def release_db_connection(conn):
    connection_pool.putconn(conn)
```

**Solución 2: Implementar Circuit Breaker**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            self.state = 'CLOSED'
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            raise e
```

### PROBLEMA: SSL/TLS Certificate Errors

**Síntomas:**
- Error: "certificate verify failed"
- Error: "SSL handshake failed"
- Browsers showing security warnings

**Diagnóstico:**
```bash
# Verificar certificado
openssl s_client -connect example.com:443 -servername example.com

# Verificar fechas de expiración
echo | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates

# Verificar cadena de certificados
openssl verify -CAfile ca-bundle.crt certificate.crt
```

**Soluciones:**

**Solución 1: Renovar Certificado con Let's Encrypt**
```bash
# Installar certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtener certificado
sudo certbot --nginx -d example.com

# Configurar renovación automática
sudo crontab -e
# Agregar línea:
0 12 * * * /usr/bin/certbot renew --quiet
```

**Solución 2: Configurar Nginx para SSL**
```nginx
server {
    listen 443 ssl http2;
    server_name example.com;
    
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
}
```

## Problemas de Deployment

### PROBLEMA: Docker containers failing to start

**Síntomas:**
- Container exits immediately
- Error: "standard_init_linux.go: exec user process caused: no such file or directory"
- Health checks failing

**Diagnóstico:**
```bash
# Verificar logs del container
docker logs container_name

# Inspeccionar container
docker inspect container_name

# Ejecutar shell en container para debug
docker run -it --entrypoint /bin/sh image_name

# Verificar recursos del sistema
docker stats
```

**Soluciones:**

**Solución 1: Corregir Dockerfile**
```dockerfile
# Problemático
FROM node:alpine
COPY . .
RUN npm install
CMD ["node", "app.js"]

# Correcto
FROM node:16-alpine

# Crear usuario no-root
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

WORKDIR /app

# Copiar archivos de dependencias primero
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copiar código fuente
COPY --chown=nextjs:nodejs . .

USER nextjs

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["node", "app.js"]
```

**Solución 2: Configurar Docker Compose con limits**
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### PROBLEMA: Kubernetes Pods in CrashLoopBackOff

**Síntomas:**
- Pods reiniciando constantemente
- Status: CrashLoopBackOff
- Application not accessible

**Diagnóstico:**
```bash
# Verificar status de pods
kubectl get pods

# Verificar logs
kubectl logs pod-name --previous

# Describir pod para eventos
kubectl describe pod pod-name

# Verificar recursos del nodo
kubectl top nodes
kubectl top pods
```

**Soluciones:**

**Solución 1: Ajustar Resource Limits**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: my-app:latest
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        # Probes para verificar salud
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Solución 2: Configurar Startup Probe para aplicaciones lentas**
```yaml
containers:
- name: app
  image: my-app:latest
  startupProbe:
    httpGet:
      path: /startup
      port: 3000
    failureThreshold: 30
    periodSeconds: 10
  livenessProbe:
    httpGet:
      path: /health
      port: 3000
    periodSeconds: 10
  readinessProbe:
    httpGet:
      path: /ready
      port: 3000
    periodSeconds: 5
```

## Herramientas de Monitoreo y Debug

### Comandos Útiles para Diagnostico

**System Resources:**
```bash
# CPU y memoria
htop
# o
top -p `pgrep -d',' node`

# Espacio en disco
df -h
du -sh /var/log/*

# Conexiones de red
netstat -tulpn | grep :3000
ss -tulpn | grep :3000

# Procesos que usan más memoria
ps aux --sort=-%mem | head

# I/O de disco
iotop
```

**Application Logs:**
```bash
# Seguir logs en tiempo real
tail -f /var/log/app/application.log

# Buscar errores en logs
grep -i error /var/log/app/application.log | tail -20

# Contar errores por hora
grep -i error /var/log/app/application.log | awk '{print $1" "$2}' | sort | uniq -c

# Analizar logs con jq (para JSON logs)
cat app.log | jq 'select(.level == "error")'
```

### Scripts de Monitoreo Automatizado

**Health Check Script:**
```bash
#!/bin/bash

# Script de health check automático
ENDPOINT="http://localhost:3000/health"
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $ENDPOINT)
    
    if [ $RESPONSE -eq 200 ]; then
        echo "$(date): Service is healthy"
        exit 0
    else
        echo "$(date): Health check failed with code $RESPONSE"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 5
    fi
done

echo "$(date): Service is unhealthy after $MAX_RETRIES attempts"
# Aquí se puede agregar lógica para restart automático
exit 1
```

---

**Nota**: Esta guía debe actualizarse regularmente basándose en los problemas más frecuentes encontrados en producción. Para reportar nuevos problemas o sugerir mejoras, crear un ticket en el sistema de seguimiento de issues.