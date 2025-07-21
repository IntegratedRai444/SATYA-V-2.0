import React from 'react';
import { render, screen } from '@testing-library/react';
import Dashboard from '../pages/Dashboard';
 
test('renders Dashboard with Scan History and Analytics', () => {
  render(<Dashboard />);
  expect(screen.getByText(/Scan History/i)).toBeInTheDocument();
  expect(screen.getByText(/Scan Analytics/i)).toBeInTheDocument();
}); 