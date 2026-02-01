// WebSocket hooks exports
export { useBaseWebSocket } from './useBaseWebSocket';
export { useWebSocket, useJobProgress } from './useWebSocket';
export { useDashboardWebSocket } from './useDashboardWebSocket';
export { useNotificationsWebSocket } from './useNotificationsWebSocket';
export { useSystemAlertsWebSocket } from './useSystemAlertsWebSocket';

// Type exports
export type { WebSocketMessage, BaseWebSocketOptions } from './useBaseWebSocket';
export type { JobProgress } from './useWebSocket';
export type { DashboardUpdateMessage } from './useDashboardWebSocket';
export type { NotificationMessage } from './useNotificationsWebSocket';
export type { SystemAlertMessage } from './useSystemAlertsWebSocket';
