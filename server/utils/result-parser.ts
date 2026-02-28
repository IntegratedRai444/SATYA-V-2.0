import { logger } from '../config/logger';

/**
 * Standard result interface for tasks.result JSONB field
 */
export interface StandardAnalysisResult {
  prediction: 'real' | 'fake' | 'inconclusive';
  confidence: number;
  reasoning?: string;
  is_deepfake?: boolean;
  model_name?: string;
  model_version?: string;
  processing_time?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Safe JSON wrapper for JSONB columns
 * Ensures all writes to JSONB columns are valid JSON objects
 */
export function safeJson(value: any): any {
  if (value === null || value === undefined) return null;
  if (typeof value === 'object') return value;
  
  // Handle status strings specially
  if (typeof value === 'string') {
    // Check if it's a known status string
    const knownStatuses = ['processing', 'completed', 'failed', 'cancelled', 'pending'];
    if (knownStatuses.includes(value.toLowerCase())) {
      logger.warn('[JSON GUARD] Auto-wrapping status string into JSON object', { 
        status: value 
      });
      return { status: value };
    }
    
    // For other strings, wrap generically
    logger.warn('[JSON GUARD] Auto-wrapping string value into JSON object', { 
      value: value.substring(0, 100) 
    });
    return { value };
  }
  
  // For other primitive types, wrap in object
  logger.warn('[JSON GUARD] Auto-wrapping primitive value into JSON object', { 
    type: typeof value,
    value: String(value).substring(0, 100)
  });
  return { value };
}

/**
 * JSONB guard rail for runtime validation
 * Prevents invalid JSONB writes with logging
 */
export function assertJsonbSafe(value: any): any {
  if (typeof value === 'string') {
    logger.warn('[JSONB GUARD] Prevented invalid JSONB write', { 
      type: 'string',
      value: value.substring(0, 100)
    });
    return { status: value };
  }
  
  if (value === null || value === undefined) {
    return null;
  }
  
  if (typeof value === 'object') {
    return value;
  }
  
  // For other primitive types, wrap safely
  logger.warn('[JSONB GUARD] Auto-wrapping primitive for JSONB', { 
    type: typeof value,
    value: String(value).substring(0, 100)
  });
  return { value };
}

/**
 * Safely parse analysis result from tasks.result JSONB field
 * Handles various result formats and provides safe defaults
 */
export function parseAnalysisResult(result: unknown): StandardAnalysisResult {
  // Handle null/undefined
  if (!result) {
    return {
      prediction: 'inconclusive',
      confidence: 0,
      reasoning: 'No result available'
    };
  }

  // Handle string results (legacy format)
  if (typeof result === 'string') {
    switch (result.toLowerCase()) {
      case 'completed':
        return {
          prediction: 'inconclusive',
          confidence: 0.5,
          reasoning: 'Analysis completed without detailed result'
        };
      case 'failed':
        return {
          prediction: 'inconclusive',
          confidence: 0,
          reasoning: 'Analysis failed'
        };
      case 'processing':
        return {
          prediction: 'inconclusive',
          confidence: 0,
          reasoning: 'Analysis still processing'
        };
      default:
        return {
          prediction: 'inconclusive',
          confidence: 0,
          reasoning: `Unknown status: ${result}`
        };
    }
  }

  // Handle object results
  if (typeof result === 'object' && result !== null) {
    const resultObj = result as Record<string, unknown>;
    
    // Extract prediction with fallback
    let prediction: 'real' | 'fake' | 'inconclusive' = 'inconclusive';
    if (typeof resultObj.prediction === 'string') {
      const pred = resultObj.prediction.toLowerCase();
      if (pred === 'real' || pred === 'fake' || pred === 'inconclusive') {
        prediction = pred;
      }
    } else if (typeof resultObj.is_deepfake === 'boolean') {
      prediction = resultObj.is_deepfake ? 'fake' : 'real';
    }

    // Extract confidence with fallback
    let confidence = 0;
    if (typeof resultObj.confidence === 'number' && 
        resultObj.confidence >= 0 && 
        resultObj.confidence <= 1) {
      confidence = resultObj.confidence;
    } else if (typeof resultObj.confidence_score === 'number' && 
               resultObj.confidence_score >= 0 && 
               resultObj.confidence_score <= 1) {
      confidence = resultObj.confidence_score;
    }

    // Extract reasoning
    let reasoning = '';
    if (typeof resultObj.reasoning === 'string') {
      reasoning = resultObj.reasoning;
    } else if (typeof resultObj.error_message === 'string') {
      reasoning = resultObj.error_message;
    } else if (typeof resultObj.summary === 'string') {
      reasoning = resultObj.summary;
    }

    return {
      prediction,
      confidence,
      reasoning: reasoning || undefined,
      is_deepfake: prediction === 'fake',
      model_name: typeof resultObj.model_name === 'string' ? resultObj.model_name : undefined,
      model_version: typeof resultObj.model_version === 'string' ? resultObj.model_version : undefined,
      processing_time: typeof resultObj.processing_time === 'number' ? resultObj.processing_time : undefined,
      metadata: typeof resultObj.metadata === 'object' && resultObj.metadata !== null 
        ? resultObj.metadata as Record<string, unknown> 
        : undefined
    };
  }

  // Handle any other type as inconclusive
  logger.warn('Unexpected result type', { 
    resultType: typeof result,
    result: String(result).substring(0, 100)
  });

  return {
    prediction: 'inconclusive',
    confidence: 0,
    reasoning: `Invalid result type: ${typeof result}`
  };
}

/**
 * Generate a unique report code for analysis results
 */
export function generateReportCode(): string {
  const date = new Date();
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const dateStr = `${year}${month}${day}`;
  
  // Generate a random 4-digit sequence
  const sequence = Math.floor(1000 + Math.random() * 9000);
  
  return `SATYA-${dateStr}-${sequence}`;
}

/**
 * Map task record to legacy scan format for backward compatibility
 */
export function mapTaskToLegacyScan(task: {
  id: string;
  type: string;
  filename: string;
  result: unknown;
  confidence_score: number;
  detection_details: unknown;
  created_at: string;
  updated_at?: string;
}) {
  const parsedResult = parseAnalysisResult(task.result);
  
  return {
    id: task.id,
    filename: task.filename,
    type: task.type,
    result: task.result,
    confidence_score: task.confidence_score,
    detection_details: task.detection_details,
    created_at: task.created_at,
    updated_at: task.updated_at,
    // Backward compatibility fields
    prediction: parsedResult.prediction,
    confidence: parsedResult.confidence,
    reasoning: parsedResult.reasoning,
    is_deepfake: parsedResult.is_deepfake,
    report_code: generateReportCode()
  };
}
