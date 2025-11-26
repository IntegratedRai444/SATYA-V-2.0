/**
 * PM2 Ecosystem Configuration
 * For production deployment with PM2 process manager
 */

module.exports = {
  apps: [
    {
      name: 'satyaai-server',
      script: './dist/index.js',
      instances: 'max',
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      },
      error_file: './logs/server-error.log',
      out_file: './logs/server-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      max_memory_restart: '1G',
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '10s',
      listen_timeout: 10000,
      kill_timeout: 5000,
      wait_ready: true,
      shutdown_with_message: true
    },
    {
      name: 'satyaai-python',
      script: 'gunicorn',
      args: '-c server/python/gunicorn.conf.py server.python.app:app',
      interpreter: 'python',
      instances: 2,
      exec_mode: 'cluster',
      env: {
        FLASK_ENV: 'production',
        PORT: 5001
      },
      error_file: './logs/python-error.log',
      out_file: './logs/python-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      max_memory_restart: '2G',
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '10s'
    }
  ],

  deploy: {
    production: {
      user: 'deploy',
      host: 'your-server.com',
      ref: 'origin/main',
      repo: 'git@github.com:yourusername/satyaai.git',
      path: '/var/www/satyaai',
      'post-deploy': 'npm install && npm run build && pm2 reload ecosystem.config.js --env production',
      'pre-setup': 'apt-get install git'
    }
  }
};
