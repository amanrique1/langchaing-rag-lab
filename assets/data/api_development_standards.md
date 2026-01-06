# API Development Standards

## RESTful Design

### REST Principles
REST (Representational State Transfer) is an architectural style for web services that defines the following constraints:

1.  **Client-Server**: Clear separation of responsibilities.
2.  **Stateless**: Every request must contain all the necessary information.
3.  **Cacheable**: Responses must be marked as cacheable or not.
4.  **Uniform Interface**: Consistent use of HTTP methods and URIs.
5.  **Layered System**: The architecture can be composed of multiple layers.

### HTTP Methods and Their Usage

| Method | Purpose | Idempotent | Safe |
|---|---|---|---|
| GET | Retrieve resources | Yes | Yes |
| POST | Create resources | No | No |
| PUT | Update/create resources | Yes | No |
| PATCH | Partial update | No | No |
| DELETE | Delete resources | Yes | No |

### URL Conventions

**Base Structure:**
```
https://api.example.com/v1/users/{id}/orders/{order_id}
```

**Rules:**
- Use plural nouns: `/users` not `/user`
- Use kebab-case for compound URLs: `/public-comments`
- Avoid verbs in URLs: `/users` not `/get-users`
- Use numbers for versioning: `/v1/`, `/v2/`

## HTTP Status Codes

### Success Codes (2xx)
-   **200 OK**: Successful request
-   **201 Created**: Resource created successfully
-   **202 Accepted**: Request accepted for processing
-   **204 No Content**: Successful operation with no response content

### Client Error Codes (4xx)
-   **400 Bad Request**: Malformed request
-   **401 Unauthorized**: Authentication required
-   **403 Forbidden**: Access denied
-   **404 Not Found**: Resource not found
-   **409 Conflict**: Conflict with the current state of the resource
-   **422 Unprocessable Entity**: Invalid input data

### Server Error Codes (5xx)
-   **500 Internal Server Error**: Internal server error
-   **502 Bad Gateway**: Gateway error
-   **503 Service Unavailable**: Service unavailable
-   **504 Gateway Timeout**: Gateway timeout

## Authentication and Authorization

### JWT (JSON Web Tokens)
Structure of a JWT:
```
Header.Payload.Signature
```

**Header Example:**
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Payload Example:**
```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022,
  "exp": 1516325422
}
```

### Authentication Implementation
1.  **Bearer Token**: `Authorization: Bearer <token>`
2.  **API Keys**: `X-API-Key: <key>`
3.  **OAuth 2.0**: For delegated authentication

## Error Handling

### Error Response Structure
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The submitted data is not valid",
    "details": [
      {
        "field": "email",
        "message": "The email format is invalid"
      },
      {
        "field": "age",
        "message": "Age must be greater than 0"
      }
    ],
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### Error Best Practices
- Provide specific error codes
- Include descriptive messages in the user's language
- Add specific details about problematic fields
- Include a `request_id` for tracking
- Maintain consistency in the structure

## Pagination

### Offset-based Pagination
```
GET /api/v1/users?offset=20&limit=10
```

**Response:**
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

### Cursor-based Pagination
```
GET /api/v1/users?cursor=eyJpZCI6MTAwfQ&limit=10
```

**Advantages of cursor:**
- Consistency in paginated results
- Better performance for large datasets
- Avoids duplicates during pagination

## Filtering and Searching

### Query Parameters
```
GET /api/v1/products?category=electronics&price_min=100&price_max=500&sort_by=price&order=asc
```

### Text Search
```
GET /api/v1/articles?q=machine+learning&fields=title,content
```

## API Versioning

### Versioning Strategies
1.  **URL Path**: `/v1/users`, `/v2/users`
2.  **Query Parameter**: `/users?version=2`
3.  **Header**: `API-Version: 2`
4.  **Accept Header**: `Accept: application/vnd.api+json;version=2`

### Deprecation Policies
- Maintain previous versions for at least 6 months
- Notify about deprecation 3 months in advance
- Include deprecation headers in responses
- Provide migration guides

## Rate Limiting

### Implementation
Response headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

### Rate Limiting Strategies
-   **Fixed Window**: Fixed limit per time window
-   **Sliding Window**: More precise sliding window
-   **Token Bucket**: Allows controlled bursts
-   **Leaky Bucket**: Constant flow of requests

## Documentation

### OpenAPI Specification
Use OpenAPI 3.0+ to document APIs:
```yaml
openapi: 3.0.0
info:
  title: User Management API
  version: 1.0.0
paths:
  /users:
    get:
      summary: Get a list of users
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
```

### Documentation Tools
-   **Swagger UI**: Interactive interface
-   **Redoc**: Elegant static documentation
-   **Postman Collections**: Exportable collections