// Global type declarations for missing modules
declare module 'jsonwebtoken' {
  export function verify(
    token: string,
    secretOrPublicKey?: string | Buffer,
    options?: { 
      algorithms?: string[]; 
      clockTolerance?: number; 
      audience?: string; 
      issuer?: string; 
      jwtid?: string; 
      keyid?: string; 
      subject?: string 
    }
  ): unknown;
}

declare module 'compression' {
  export interface CompressionOptions {
    level?: number;
    chunkSize?: number;
    memLevel?: number;
    strategy?: number;
    threshold?: number;
    windowBits?: number;
    dictionary?: Buffer | string;
  }
  export function createFilter(options?: CompressionOptions): unknown;
  export function filter(req: unknown, res: unknown, options?: CompressionOptions): unknown;
  export function compress(data: Buffer | string, options?: CompressionOptions): Buffer;
}

declare module 'pdfkit' {
  export interface PDFDocument {
    info: {
      Title?: string;
      Author?: string;
      Subject?: string;
      Creator?: string;
      Producer?: string;
      CreationDate?: Date;
      ModDate?: Date;
      Keywords?: string[];
    };
  }
  export interface PDFKit {
    document: PDFDocument;
    writePdf: (filename: string, data: PDFDocument, options?: Record<string, unknown>) => Promise<void>;
    getDocumentInfo: (filename: string) => Promise<PDFDocument>;
    getFormFields: (filename: string) => Promise<unknown[]>;
    getAcroForm: (filename: string) => Promise<unknown>;
    getAcroFormFields: (filename: string) => Promise<unknown[]>;
    getRawTextContent: (filename: string) => Promise<string>;
  }
}

declare module 'ioredis' {
  export interface RedisCommand {
    args: string[];
    callback: (err: Error | null, reply: unknown) => void;
  }
  export interface Redis {
    get(key: string, callback: (err: Error | null, reply: unknown) => void): void;
    set(key: string, value: string, callback?: (err: Error | null, reply: unknown) => void): void;
    setex(key: string, value: string, mode: string, ttl?: number, callback?: (err: Error | null, reply: unknown) => void): void;
    del(key: string, callback?: (err: Error | null, reply: unknown) => void): void;
    exists(key: string, callback: (err: Error | null, reply: unknown) => void): void;
    expire(key: string, seconds: number, callback?: (err: Error | null, reply: unknown) => void): void;
    expireat(key: string, timestamp: string, callback?: (err: Error | null, reply: unknown) => void): void;
    ttl(key: string, callback?: (err: Error | null, reply: unknown) => void): void;
    keys(pattern: string, callback: (err: Error | null, reply: unknown) => void): void;
    flushdb(callback?: (err: Error | null, reply: unknown) => void): void;
    flushall(callback?: (err: Error | null, reply: unknown) => void): void;
    multi: RedisCommand[];
    scan(command: string, ...args: RedisCommand[], callback?: (err: Error | null, reply: unknown) => void): void;
    select(pattern: string, ...args: RedisCommand[], callback?: (err: Error | null, reply: unknown) => void): void;
    selectall(pattern: string, callback?: (err: Error | null, reply: unknown) => void): void;
  }
}

declare module 'axios' {
  export interface AxiosInstance {
    (config?: unknown): unknown;
    delete(url: string, config?: unknown): unknown;
    get(url: string, config?: unknown): unknown;
    head(url: string, config?: unknown): unknown;
    options(url: string, config?: unknown): unknown;
    post(url: string, data?: unknown, config?: unknown): unknown;
    put(url: string, data?: unknown, config?: unknown): unknown;
    patch(url: string, data?: unknown, config?: unknown): unknown;
    postForm(url: string, data?: unknown, config?: unknown): unknown;
    putForm(url: string, data?: unknown, config?: unknown): unknown;
    request(config?: unknown): unknown;
    interceptors: unknown[];
    defaults: unknown;
    getUri: (url?: string, config?: unknown) => string;
    postUri: (url?: string, config?: unknown) => string;
  }
}

declare module 'bcryptjs' {
  export function compare(password: string, hash: string): Promise<boolean>;
  export function genSalt(rounds?: number): Promise<string>;
  export function hash(password: string, salt?: string, rounds?: number, callback?: (err: Error | null, progress?: number, key: Buffer) => void): void;
  export function getRounds(hash: string): number;
}

declare module 'prom-client' {
  interface Registry {
    get: (key: string) => Promise<string>;
    set: (key: string, value: string) => Promise<void>;
    del: (key: string) => Promise<void>;
    exists: (key: string) => Promise<boolean>;
    keys: (pattern: string) => Promise<string[]>;
    rename: (oldKey: string, newKey: string) => Promise<void>;
    clear: (pattern?: string) => Promise<void>;
    getMemoryUsage: () => Promise<{
      rss: number;
      heapTotal: number;
      heapUsed: number;
      external: number;
      arrayBuffers: number;
      v8: {
        rss: number;
        heapTotal: number;
        heapUsed: number;
        external: number;
        arrayBuffers: number;
      };
    }>;
  }
}

declare module 'check-disk-space' {
  function checkSync(path: string): number;
  function check(path: string, callback?: (err: Error | null, used?: number, available?: number) => void): void;
}

declare module 'openai' {
  interface CreateChatCompletionResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: Array<{
      text: string;
      index: number;
      logprobs: number;
      finish_reason: string;
    }>;
  }
}

declare module 'abort-controller' {
  interface AbortSignal {
    aborted: boolean;
    reason: unknown;
    onabort: ((reason: unknown) => void);
  }
  
  interface AbortController {
    abort(): AbortSignal;
  }
}

declare module 'isomorphic-dompurify' {
  interface DOMPurify {
    sanitize(dirty: string): string;
    isSupported(name: string): boolean;
  }
}
