#!/bin/bash

# Test authentication with curl
curl -X POST http://localhost:5001/api/v2/analysis/image \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ0YnBiZ2hjZWJ3Z3pxZnNnbXhrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgwNTc2MTcsImV4cCI6MjA4MzYzMzYxNzA0LjA0In0" \
  -F "file=@test.txt" \
  -v
