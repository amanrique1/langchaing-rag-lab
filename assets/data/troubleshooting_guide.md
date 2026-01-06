# Troubleshooting Guide - Common Problems

## Performance Issues

### PROBLEM: High Latency in API Responses

**Symptoms:**
- Response times > 2 seconds
- Frequent timeouts
- User complaints about slowness

**Diagnosis:**
1.  Check APM (Application Performance Monitoring) metrics.
2.  Analyze application and database logs.
3.  Review CPU and memory utilization.
4.  Check for slow database queries.

**Solutions:**

**Solution 1: Query Optimization**
```sql
-- Before (slow query)
SELECT * FROM users u 
JOIN orders o ON u.id = o.user_id 
WHERE u.active = 1;

-- After (optimized query with indexes)
SELECT u.id, u.name, o.total 
FROM users u 
JOIN orders o ON u.id = o.user_id 
WHERE u.active = 1 
AND u.creation_date > '2024-01-01';

-- Add index
CREATE INDEX idx_users_active_date ON users(active, creation_date);
```

**Solution 2: Implement Caching**
```python
import redis
from functools import wraps

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and save to cache
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

**Solution 3: Database Connection Pool**
```python
# Optimized connection settings
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}
```

### PROBLEM: Memory Leaks in Node.js Application

**Symptoms:**
- Memory usage grows constantly
- Performance degrades over time
- Eventual crash due to out of memory

**Diagnosis:**
```bash
# Monitor memory usage
node --inspect --max-old-space-size=4096 app.js

# Generate heap dump
kill -USR2 <node_process_id>

# Analyze with tools
node --prof app.js
node --prof-process isolate-0x[...].log > profile.txt
```

**Solutions:**

**Solution 1: Clean Up Event Listeners**
```javascript
// Problematic
class MyComponent {
    constructor() {
        window.addEventListener('resize', this.handleResize);
    }
}

// Correct
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

**Solution 2: Stream Management**
```javascript
// Problematic
const fs = require('fs');
const stream = fs.createReadStream('large-file.txt');
// Stream is never closed

// Correct
const fs = require('fs');
const stream = fs.createReadStream('large-file.txt');
stream.on('end', () => stream.close());
stream.on('error', () => stream.close());
```

## Connectivity Issues

### PROBLEM: Database Connection Errors

**Symptoms:**
- Error: "Connection refused"
- Error: "Too many connections"
- Intermittent timeouts

**Diagnosis:**
```bash
# Check connectivity
telnet db-server 5432

# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Review PostgreSQL logs
tail -f /var/log/postgresql/postgresql.log
```

**Solutions:**

**Solution 1: Configure Connection Pooling**
```python
import psycopg2.pool

# Create connection pool
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

**Solution 2: Implement Circuit Breaker**
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

### PROBLEM: SSL/TLS Certificate Errors

**Symptoms:**
- Error: "certificate verify failed"
- Error: "SSL handshake failed"
- Browsers showing security warnings

**Diagnosis:**
```bash
# Verify certificate
openssl s_client -connect example.com:443 -servername example.com

# Check expiration dates
echo | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates

# Verify certificate chain
openssl verify -CAfile ca-bundle.crt certificate.crt
```

**Solutions:**

**Solution 1: Renew Certificate with Let's Encrypt**
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d example.com

# Configure automatic renewal
sudo crontab -e
# Add line:
0 12 * * * /usr/bin/certbot renew --quiet
```

**Solution 2: Configure Nginx for SSL**
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

## Deployment Issues

### PROBLEM: Docker containers failing to start

**Symptoms:**
- Container exits immediately
- Error: "standard_init_linux.go: exec user process caused: no such file or directory"
- Health checks failing

**Diagnosis:**
```bash
# Check container logs
docker logs container_name

# Inspect container
docker inspect container_name

# Run a shell in the container for debugging
docker run -it --entrypoint /bin/sh image_name

# Check system resources
docker stats
```

**Solutions:**

**Solution 1: Correct the Dockerfile**
```dockerfile
# Problematic
FROM node:alpine
COPY . .
RUN npm install
CMD ["node", "app.js"]

# Correct
FROM node:16-alpine

# Create a non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

WORKDIR /app

# Copy dependency files first
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY --chown=nextjs:nodejs . .

USER nextjs

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["node", "app.js"]
```

**Solution 2: Configure Docker Compose with limits**
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

### PROBLEM: Kubernetes Pods in CrashLoopBackOff

**Symptoms:**
- Pods restarting constantly
- Status: CrashLoopBackOff
- Application not accessible

**Diagnosis:**
```bash
# Check pod status
kubectl get pods

# Check logs (including previous container instance)
kubectl logs pod-name --previous

# Describe pod for events
kubectl describe pod pod-name

# Check node and pod resources
kubectl top nodes
kubectl top pods
```

**Solutions:**

**Solution 1: Adjust Resource Limits**
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
        # Probes to check health
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

**Solution 2: Configure Startup Probe for slow-starting applications**
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

## Monitoring and Debugging Tools

### Useful Diagnostic Commands

**System Resources:**
```bash
# CPU and memory
htop
# or
top -p `pgrep -d',' node`

# Disk space
df -h
du -sh /var/log/*

# Network connections
netstat -tulpn | grep :3000
ss -tulpn | grep :3000

# Processes using the most memory
ps aux --sort=-%mem | head

# Disk I/O
iotop
```

**Application Logs:**
```bash
# Follow logs in real-time
tail -f /var/log/app/application.log

# Search for errors in logs
grep -i error /var/log/app/application.log | tail -20

# Count errors per hour
grep -i error /var/log/app/application.log | awk '{print $1" "$2}' | sort | uniq -c

# Analyze logs with jq (for JSON logs)
cat app.log | jq 'select(.level == "error")'
```

### Automated Monitoring Scripts

**Health Check Script:**
```bash
#!/bin/bash

# Automatic health check script
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
# Logic for automatic restart can be added here
exit 1
```

---

**Note**: This guide should be updated regularly based on the most frequent problems encountered in production. To report new problems or suggest improvements, create a ticket in the issue tracking system.