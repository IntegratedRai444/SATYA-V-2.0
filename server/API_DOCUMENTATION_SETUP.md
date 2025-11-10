# API Documentation Setup Guide

## Overview

This guide will help you add Swagger/OpenAPI documentation to your SatyaAI backend.

## Installation

```bash
npm install swagger-jsdoc swagger-ui-express
npm install --save-dev @types/swagger-jsdoc @types/swagger-ui-express
```

## Setup Steps

### 1. Create Swagger Configuration

Create `server/config/swagger.ts`:

```typescript
import swaggerJsdoc from 'swagger-jsdoc';

const options: swaggerJsdoc.Options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'SatyaAI API Documentation',
      version: '2.0.0',
      description: 'Deepfake Detection API - Comprehensive documentation for all endpoints',
      contact: {
        name: 'SatyaAI Support',
        email: 'support@satyaai.com'
      },
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT'
      }
    },
    servers: [
      {
        url: 'http://localhost:3000',
        description: 'Development server'
      },
      {
        url: 'https://api.satyaai.com',
        description: 'Production server'
      }
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT'
        }
      }
    },
    security: [{
      bearerAuth: []
    }]
  },
  apis: ['./server/routes/*.ts', './server/routes/**/*.ts']
};

export const swaggerSpec = swaggerJsdoc(options);
```

### 2. Add to Server

In `server/index.ts`, add:

```typescript
import swaggerUi from 'swagger-ui-express';
import { swaggerSpec } from './config/swagger';

// Add after other middleware
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: 'SatyaAI API Docs'
}));

// Serve swagger spec as JSON
app.get('/api-docs.json', (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.send(swaggerSpec);
});
```

### 3. Document Your Routes

Add JSDoc comments to your route files:

#### Example: Authentication Routes

```typescript
/**
 * @swagger
 * /api/auth/login:
 *   post:
 *     summary: User login
 *     tags: [Authentication]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - password
 *             properties:
 *               email:
 *                 type: string
 *                 format: email
 *                 example: user@example.com
 *               password:
 *                 type: string
 *                 format: password
 *                 example: SecurePassword123!
 *     responses:
 *       200:
 *         description: Login successful
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 token:
 *                   type: string
 *                   example: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
 *                 user:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     email:
 *                       type: string
 *                     name:
 *                       type: string
 *       401:
 *         description: Invalid credentials
 *       429:
 *         description: Too many login attempts
 */
router.post('/login', authRateLimit, async (req, res) => {
  // Implementation
});
```

#### Example: Analysis Routes

```typescript
/**
 * @swagger
 * /api/analyze/image:
 *   post:
 *     summary: Analyze image for deepfakes
 *     tags: [Analysis]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             required:
 *               - file
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *                 description: Image file to analyze
 *               options:
 *                 type: object
 *                 properties:
 *                   advanced:
 *                     type: boolean
 *                     default: false
 *                   threshold:
 *                     type: number
 *                     minimum: 0
 *                     maximum: 1
 *                     default: 0.5
 *     responses:
 *       200:
 *         description: Analysis complete
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 scanId:
 *                   type: string
 *                 result:
 *                   type: object
 *                   properties:
 *                     isDeepfake:
 *                       type: boolean
 *                     confidence:
 *                       type: number
 *                     details:
 *                       type: object
 *       400:
 *         description: Invalid file or parameters
 *       401:
 *         description: Unauthorized
 *       429:
 *         description: Rate limit exceeded
 */
router.post('/image', analysisRateLimit, authenticate, async (req, res) => {
  // Implementation
});
```

### 4. Common Schemas

Define reusable schemas:

```typescript
/**
 * @swagger
 * components:
 *   schemas:
 *     Error:
 *       type: object
 *       properties:
 *         success:
 *           type: boolean
 *           example: false
 *         error:
 *           type: string
 *           example: Error message
 *         code:
 *           type: string
 *           example: ERROR_CODE
 *     
 *     AnalysisResult:
 *       type: object
 *       properties:
 *         scanId:
 *           type: string
 *         isDeepfake:
 *           type: boolean
 *         confidence:
 *           type: number
 *           minimum: 0
 *           maximum: 1
 *         timestamp:
 *           type: string
 *           format: date-time
 *         details:
 *           type: object
 */
```

## Access Documentation

Once set up, access your API documentation at:

- **Swagger UI:** `http://localhost:3000/api-docs`
- **JSON Spec:** `http://localhost:3000/api-docs.json`

## Documentation Checklist

- [ ] Install dependencies
- [ ] Create swagger config
- [ ] Add to server
- [ ] Document auth routes
- [ ] Document analysis routes
- [ ] Document upload routes
- [ ] Document dashboard routes
- [ ] Document health routes
- [ ] Add common schemas
- [ ] Test documentation UI
- [ ] Add examples for all endpoints
- [ ] Document error responses
- [ ] Add authentication examples

## Best Practices

1. **Keep docs in sync** - Update docs when changing routes
2. **Add examples** - Include realistic request/response examples
3. **Document errors** - Show all possible error responses
4. **Use schemas** - Define reusable schemas for common objects
5. **Add descriptions** - Explain what each endpoint does
6. **Security** - Document authentication requirements
7. **Rate limits** - Mention rate limiting in descriptions

## Benefits

✅ Interactive API testing  
✅ Automatic documentation generation  
✅ Client SDK generation  
✅ Better developer experience  
✅ Reduced support requests  
✅ Easier onboarding  

## Next Steps

1. Install dependencies
2. Create swagger config file
3. Add to server/index.ts
4. Start documenting routes (begin with auth)
5. Test at http://localhost:3000/api-docs
6. Iterate and expand coverage
