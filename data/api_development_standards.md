# Estándares de Desarrollo de APIs

## Diseño RESTful

### Principios REST
REST (Representational State Transfer) es un estilo arquitectónico para servicios web que define las siguientes restricciones:

1. **Cliente-Servidor**: Separación clara de responsabilidades
2. **Sin Estado**: Cada solicitud debe contener toda la información necesaria
3. **Cacheable**: Las respuestas deben ser marcadas como cacheables o no
4. **Interfaz Uniforme**: Uso consistente de métodos HTTP y URIs
5. **Sistema en Capas**: La arquitectura puede estar compuesta por múltiples capas

### Métodos HTTP y su Uso

| Método | Propósito | Idempotente | Seguro |
|--------|-----------|-------------|--------|
| GET | Obtener recursos | Sí | Sí |
| POST | Crear recursos | No | No |
| PUT | Actualizar/crear recursos | Sí | No |
| PATCH | Actualización parcial | No | No |
| DELETE | Eliminar recursos | Sí | No |

### Convenciones de URL

**Estructura Base:**
```
https://api.ejemplo.com/v1/usuarios/{id}/pedidos/{pedido_id}
```

**Reglas:**
- Usar sustantivos en plural: `/usuarios` no `/usuario`
- Usar kebab-case para URLs compuestas: `/comentarios-publicos`
- Evitar verbos en las URLs: `/usuarios` no `/obtener-usuarios`
- Usar números para versionado: `/v1/`, `/v2/`

## Códigos de Estado HTTP

### Códigos de Éxito (2xx)
- **200 OK**: Solicitud exitosa
- **201 Created**: Recurso creado exitosamente
- **202 Accepted**: Solicitud aceptada para procesamiento
- **204 No Content**: Operación exitosa sin contenido de respuesta

### Códigos de Error del Cliente (4xx)
- **400 Bad Request**: Solicitud malformada
- **401 Unauthorized**: Autenticación requerida
- **403 Forbidden**: Acceso denegado
- **404 Not Found**: Recurso no encontrado
- **409 Conflict**: Conflicto con el estado actual del recurso
- **422 Unprocessable Entity**: Datos de entrada inválidos

### Códigos de Error del Servidor (5xx)
- **500 Internal Server Error**: Error interno del servidor
- **502 Bad Gateway**: Error de gateway
- **503 Service Unavailable**: Servicio no disponible
- **504 Gateway Timeout**: Timeout del gateway

## Autenticación y Autorización

### JWT (JSON Web Tokens)
Estructura de un JWT:
```
Header.Payload.Signature
```

**Ejemplo de Header:**
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Ejemplo de Payload:**
```json
{
  "sub": "1234567890",
  "name": "Juan Pérez",
  "iat": 1516239022,
  "exp": 1516325422
}
```

### Implementación de Autenticación
1. **Bearer Token**: `Authorization: Bearer <token>`
2. **API Keys**: `X-API-Key: <key>`
3. **OAuth 2.0**: Para autenticación delegada

## Manejo de Errores

### Estructura de Respuesta de Error
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Los datos enviados no son válidos",
    "details": [
      {
        "field": "email",
        "message": "El formato del email es inválido"
      },
      {
        "field": "edad",
        "message": "La edad debe ser mayor a 0"
      }
    ],
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### Mejores Prácticas para Errores
- Proporcionar códigos de error específicos
- Incluir mensajes descriptivos en idioma del usuario
- Agregar detalles específicos sobre campos problemáticos
- Incluir un request_id para rastreo
- Mantener consistencia en la estructura

## Paginación

### Paginación Basada en Offset
```
GET /api/v1/usuarios?offset=20&limit=10
```

**Respuesta:**
```json
{
  "data": [...],
  "pagination": {
    "offset": 20,
    "limit": 10,
    "total": 150,
    "has_next": true,
    "has_previous": true
  }
}
```

### Paginación Basada en Cursor
```
GET /api/v1/usuarios?cursor=eyJpZCI6MTAwfQ&limit=10
```

**Ventajas del cursor:**
- Consistencia en resultados paginados
- Mejor rendimiento para datasets grandes
- Evita duplicados durante la paginación

## Filtrado y Búsqueda

### Parámetros de Consulta
```
GET /api/v1/productos?categoria=electronica&precio_min=100&precio_max=500&ordenar_por=precio&orden=asc
```

### Búsqueda de Texto
```
GET /api/v1/articulos?q=machine+learning&campos=titulo,contenido
```

## Versionado de APIs

### Estrategias de Versionado
1. **URL Path**: `/v1/usuarios`, `/v2/usuarios`
2. **Query Parameter**: `/usuarios?version=2`
3. **Header**: `API-Version: 2`
4. **Accept Header**: `Accept: application/vnd.api+json;version=2`

### Políticas de Deprecación
- Mantener versiones anteriores por al menos 6 meses
- Notificar deprecación con 3 meses de anticipación
- Incluir headers de deprecación en respuestas
- Proporcionar guías de migración

## Rate Limiting

### Implementación
Headers de respuesta:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

### Estrategias de Rate Limiting
- **Fixed Window**: Límite fijo por ventana de tiempo
- **Sliding Window**: Ventana deslizante más precisa
- **Token Bucket**: Permite ráfagas controladas
- **Leaky Bucket**: Flujo constante de solicitudes

## Documentación

### OpenAPI Specification
Usar OpenAPI 3.0+ para documentar APIs:
```yaml
openapi: 3.0.0
info:
  title: API de Gestión de Usuarios
  version: 1.0.0
paths:
  /usuarios:
    get:
      summary: Obtener lista de usuarios
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
```

### Herramientas de Documentación
- **Swagger UI**: Interfaz interactiva
- **Redoc**: Documentación estática elegante
- **Postman Collections**: Colecciones exportables