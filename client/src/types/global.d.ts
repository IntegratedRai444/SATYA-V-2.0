/// <reference types="vite/client" />

declare global {
  namespace NodeJS {
    interface Timeout {
      ref(): void;
      unref(): void;
    }
  }
}

export {};
