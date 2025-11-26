import http from 'k6/http';
import { check, sleep } from 'k6';
import { FormData } from 'https://jslib.k6.io/formdata/0.0.2/index.js';

export const options = {
  stages: [
    { duration: '30s', target: 5 },
    { duration: '1m', target: 5 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  const fd = new FormData();
  fd.append('file', http.file('test.jpg', 'test-image-data'));
  
  const res = http.post(`${BASE_URL}/api/analysis/upload`, fd.body(), {
    headers: { 'Content-Type': 'multipart/form-data; boundary=' + fd.boundary },
  });
  
  check(res, {
    'upload successful': (r) => r.status === 200 || r.status === 201,
  });
  
  sleep(2);
}
