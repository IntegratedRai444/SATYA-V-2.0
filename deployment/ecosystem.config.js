/**
 * PM2 Ecosystem Configuration for SatyaAI
 * Production deployment with PM2 process manager
 */

module.exports = {
  apps: [
    {
      name: 'satyaai-node',
      script: 'dist/index.js',
      instances: 'max',
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'production',
        PORT: 3000,
        PYTHON_SERVER_PORT: 5001
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 3000,
        PYTHON_SERVER_PORT: 5001
      },
      // Restart configuration
      max_restarts: 10,
      min_uptime: '10s',
      max_memory_restart: '1G',
      
      // Logging
      log_file: './logs/satyaai-node.log',
      out_file: './logs/satyaai-node-out.log',
      error_file: './logs/satyaai-node-error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      
      // Monitoring
      monitoring: false,
      
      // Advanced features
      watch: false,
      ignore_watch: ['node_modules', 'logs', 'uploads'],
      
      // Health check
      health_check_grace_period: 3000,
      health_check_fatal_exceptions: true
    },
    {
      name: 'satyaai-python',
      script: 'run_satyaai.py',
      interpreter: 'python3',
      instances: 2,
      exec_mode: 'fork',
      env: {
        NODE_ENV: 'production',
        PORT: 5001,
        PYTHON_SERVER_PORT: 5001,
        FLASK_ENV: 'production'
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 5001,
        PYTHON_SERVER_PORT: 5001,
        FLASK_ENV: 'production'
      },
      
      // Restart configuration
      max_restarts: 10,
      min_uptime: '10s',
      max_memory_restart: '2G',
      
      // Logging
      log_file: './logs/satyaai-python.log',
      out_file: './logs/satyaai-python-out.log',
      error_file: './logs/satyaai-python-error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      
      // Monitoring
      monitoring: false,
      
      // Advanced features
      watch: false,
      ignore_watch: ['node_modules', 'logs', 'uploads', '__pycache__'],
      
      // Health check
      health_check_grace_period: 5000,
      health_check_fatal_exceptions: true
    }
  ],

  deploy: {
    production: {
      user: 'deploy',
      host: ['your-server.com'],
      ref: 'origin/main',
      repo: 'git@github.com:your-username/satyaai.git',
      path: '/var/www/satyaai',
      'pre-deploy-local': '',
      'post-deploy': 'npm install && npm run build && pm2 reload ecosystem.config.js --env production',
      'pre-setup': '',
      'post-setup': 'ls -la'
    }
  }
};