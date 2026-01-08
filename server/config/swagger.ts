import swaggerJsdoc, { Options } from 'swagger-jsdoc';
import { serve as swaggerUiServe, setup as swaggerUiSetup, SwaggerUiOptions } from 'swagger-ui-express';
import { Express, Request, Response, NextFunction } from 'express';

const API_VERSION = process.env.API_VERSION || '1.0.0';
const NODE_ENV = process.env.NODE_ENV || 'development';

// Configure Swagger options
const swaggerOptions: Options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'SatyaAI API',
      version: API_VERSION,
      description: 'API documentation for SatyaAI',
      contact: {
        name: 'API Support',
        url: 'https://satyaai.com/support',
        email: 'support@satyaai.com'
      },
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT',
      },
    },
    servers: [
      {
        url: NODE_ENV === 'production' 
          ? 'https://api.satyaai.com' 
          : 'http://localhost:3000',
        description: NODE_ENV === 'production' ? 'Production' : 'Development',
      },
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
          description: 'Enter JWT token in the format: Bearer <token>'
        },
      },
    },
    security: [
      {
        bearerAuth: [] as string[],
      },
    ],
  },
  apis: [
    './src/routes/*.ts',
    './src/models/*.ts',
    './src/controllers/*.ts',
  ],
};

// Initialize Swagger
const swaggerSpec = swaggerJsdoc(swaggerOptions);

// Swagger UI options
const swaggerUiOptions: SwaggerUiOptions = {
  explorer: true,
  customCss: `
    .swagger-ui .topbar { display: none }
    .swagger-ui .info { margin: 20px 0 }
  `,
  customSiteTitle: 'SatyaAI API Documentation',
  customfavIcon: '/favicon.ico',
  swaggerOptions: {
    docExpansion: 'list',
    filter: true,
    showRequestDuration: true,
  },
};

// Function to setup Swagger UI
export const setupSwagger = (app: Express): void => {
  // Serve Swagger UI at /api-docs
  app.use('/api-docs', 
    swaggerUiServe,
    (req: Request, res: Response, next: NextFunction) => {
      swaggerUiSetup(swaggerSpec, swaggerUiOptions)(req, res, next);
    }
  );

  // Serve Swagger JSON
  app.get('/api-docs.json', (_req: Request, res: Response) => {
    res.setHeader('Content-Type', 'application/json');
    res.send(swaggerSpec);
  });
};

// Export the Swagger spec and UI options
export { 
  swaggerSpec, 
  swaggerUiOptions 
};

// Export swaggerUi with proper types
export const swaggerUi = {
  serve: swaggerUiServe,
  setup: swaggerUiSetup
};
