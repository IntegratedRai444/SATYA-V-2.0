// Google Analytics gtag type definitions
declare global {
    interface Window {
        gtag?: (
            command: 'event' | 'config' | 'set' | 'get',
            targetId: string,
            config?: Record<string, any>
        ) => void;
    }
}

export { };
