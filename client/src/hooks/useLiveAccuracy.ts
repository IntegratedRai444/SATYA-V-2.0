import { useState, useEffect, useCallback, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import type { HistoryItem, ApiResponse } from '@/types/api'

interface AccuracyData {
  score: number
  basedOnAnalyses: number
  lastUpdated: Date
  isCalculating: boolean
}

interface UseLiveAccuracyOptions {
  /** Refresh interval in milliseconds (default: 5000ms) */
  refreshInterval?: number
  /** Number of recent analyses to consider (default: 10) */
  sampleSize?: number
  /** Enable/disable live updates (default: true) */
  enabled?: boolean
  /** Weighted average calculation (default: true) */
  weightedAverage?: boolean
}

const fetchRecentAnalyses = async (sampleSize: number = 10): Promise<HistoryItem[]> => {
  try {
    const response = await api.get('/history', { 
      params: { 
        limit: sampleSize,
        sortBy: 'date',
        sortOrder: 'desc'
      }
    }) as ApiResponse<{ history: HistoryItem[] }>
    
    if (!response.success || !response.data) {
      throw new Error('Failed to fetch recent analyses')
    }
    
    return response.data.history || []
  } catch (error) {
    console.error('Failed to fetch recent analyses:', error)
    throw error
  }
}

export const useLiveAccuracy = (options: UseLiveAccuracyOptions = {}) => {
  const {
    refreshInterval = 5000,
    sampleSize = 10,
    enabled = true,
    weightedAverage = true
  } = options

  // State for live accuracy
  const [accuracyData, setAccuracyData] = useState<AccuracyData>({
    score: 0,
    basedOnAnalyses: 0,
    lastUpdated: new Date(),
    isCalculating: false
  })

  // Ref to track previous score for smooth transitions
  const previousScoreRef = useRef<number>(0)
  const animationFrameRef = useRef<number>()

  // Query for recent analyses
  const { 
    data: analysesData, 
    isLoading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['recent-analyses', sampleSize],
    queryFn: () => fetchRecentAnalyses(sampleSize),
    staleTime: refreshInterval - 1000, // Slightly less than refresh interval
    refetchInterval: enabled ? refreshInterval : false,
    enabled: enabled,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  // Calculate accuracy from recent analyses
  const calculateAccuracy = useCallback((analyses: HistoryItem[]): number => {
    if (!analyses || analyses.length === 0) {
      return 0
    }

    // Filter only completed analyses with confidence scores
    const completedAnalyses = analyses.filter(
      analysis => analysis.status === 'completed' && 
                  analysis.confidence !== undefined && 
                  analysis.confidence > 0
    )

    if (completedAnalyses.length === 0) {
      return 0
    }

    const recentAnalyses = completedAnalyses.slice(0, sampleSize)
    
    if (weightedAverage) {
      // Weighted average: more recent analyses have higher weight
      let weightedSum = 0
      let totalWeight = 0
      
      recentAnalyses.forEach((analysis, index) => {
        const weight = recentAnalyses.length - index // More recent = higher weight
        const confidence = analysis.confidence || 0
        weightedSum += confidence * weight
        totalWeight += weight
      })
      
      return totalWeight > 0 ? Math.round((weightedSum / totalWeight) * 100) : 0
    } else {
      // Simple average
      const totalConfidence = recentAnalyses.reduce((sum, analysis) => sum + (analysis.confidence || 0), 0)
      return Math.round((totalConfidence / recentAnalyses.length) * 100)
    }
  }, [sampleSize, weightedAverage])

  // Smooth animation for score changes
  const animateScoreChange = useCallback((fromScore: number, toScore: number) => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }

    const duration = 800 // Animation duration in ms
    const startTime = Date.now()

    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      // Easing function for smooth animation
      const easeOutQuart = 1 - Math.pow(1 - progress, 4)
      const currentScore = Math.round(fromScore + (toScore - fromScore) * easeOutQuart)

      setAccuracyData(prev => ({
        ...prev,
        score: currentScore,
        isCalculating: progress < 1
      }))

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate)
      } else {
        setAccuracyData(prev => ({
          ...prev,
          isCalculating: false
        }))
      }
    }

    animate()
  }, [])

  // Update accuracy when data changes
  useEffect(() => {
    if (!analysesData || isLoading) return

    const newScore = calculateAccuracy(analysesData)
    const previousScore = previousScoreRef.current

    // Only animate if score changed significantly
    if (Math.abs(newScore - previousScore) > 1) {
      // Defer state update to avoid cascading renders
      setTimeout(() => {
        setAccuracyData(prev => ({ ...prev, isCalculating: true }))
      }, 0)
      animateScoreChange(previousScore, newScore)
      previousScoreRef.current = newScore
    }

    const completedAnalyses = analysesData.filter(
      analysis => analysis.status === 'completed' && 
                  analysis.confidence !== undefined && 
                  analysis.confidence > 0
    )

    // Defer state update to avoid cascading renders
    setTimeout(() => {
      setAccuracyData(prev => ({
        ...prev,
        basedOnAnalyses: completedAnalyses.slice(0, sampleSize).length,
        lastUpdated: new Date()
      }))
    }, 0)
  }, [analysesData, isLoading, calculateAccuracy, sampleSize, animateScoreChange])

  // Manual refresh function
  const refreshAccuracy = useCallback(async () => {
    setAccuracyData(prev => ({ ...prev, isCalculating: true }))
    await refetch()
  }, [refetch])

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  return {
    accuracy: accuracyData.score,
    basedOnAnalyses: accuracyData.basedOnAnalyses,
    lastUpdated: accuracyData.lastUpdated,
    isCalculating: accuracyData.isCalculating,
    isLoading,
    error,
    refresh: refreshAccuracy,
    hasData: accuracyData.basedOnAnalyses > 0,
    status: error ? 'error' : isLoading ? 'loading' : 'success'
  }
}

export type UseLiveAccuracyReturn = ReturnType<typeof useLiveAccuracy>
