import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const glowVariants = cva(
  "absolute rounded-full blur-3xl opacity-20 -z-10 transition-all duration-1000 ease-in-out",
  {
    variants: {
      variant: {
        default: "bg-cyan-400",
        blue: "bg-blue-400",
        purple: "bg-purple-400",
        emerald: "bg-emerald-400",
        teal: "bg-teal-400",
      },
      size: {
        sm: "w-32 h-32",
        md: "w-64 h-64", 
        lg: "w-96 h-96",
        xl: "w-[512px] h-[512px]",
        full: "w-full h-full",
      },
      intensity: {
        subtle: "opacity-10 blur-2xl",
        medium: "opacity-20 blur-3xl",
        strong: "opacity-30 blur-3xl",
        intense: "opacity-40 blur-3xl",
      },
      position: {
        "top-left": "top-0 left-0",
        "top-right": "top-0 right-0", 
        "bottom-left": "bottom-0 left-0",
        "bottom-right": "bottom-0 right-0",
        "center": "top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2",
        "top-center": "top-0 left-1/2 -translate-x-1/2",
        "bottom-center": "bottom-0 left-1/2 -translate-x-1/2",
        "left-center": "left-0 top-1/2 -translate-y-1/2",
        "right-center": "right-0 top-1/2 -translate-y-1/2",
      },
      animation: {
        none: "",
        pulse: "animate-pulse",
        "pulse-slow": "animate-pulse",
        breathe: "animate-pulse",
        float: "animate-bounce",
      }
    },
    defaultVariants: {
      variant: "default",
      size: "lg",
      intensity: "subtle",
      position: "center",
      animation: "none",
    },
  }
)

export interface GlowProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof glowVariants> {
  /**
   * Whether the glow should be animated
   * @default false
   */
  animated?: boolean
  /**
   * Custom animation duration in milliseconds
   * @default 1000
   */
  animationDuration?: number
  /**
   * Multiple glow layers for depth effect
   */
  layers?: Array<{
    variant?: VariantProps<typeof glowVariants>["variant"]
    size?: VariantProps<typeof glowVariants>["size"] 
    intensity?: VariantProps<typeof glowVariants>["intensity"]
    position?: VariantProps<typeof glowVariants>["position"]
    offset?: { x: number; y: number }
  }>
}

const Glow = React.forwardRef<HTMLDivElement, GlowProps>(
  ({ className, variant, size, intensity, position, animated = false, animationDuration = 1000, layers, ...props }, ref) => {
    const baseClasses = glowVariants({ variant, size, intensity, position })
    
    // Add custom animation duration if specified
    const animationStyle = animated ? {
      animationDuration: `${animationDuration}ms`,
    } : {}

    // If layers are provided, render multiple glow elements
    if (layers && layers.length > 0) {
      return (
        <div ref={ref} className={cn("relative", className)} {...props}>
          {layers.map((layer, index) => {
            const layerClasses = glowVariants({
              variant: layer.variant || variant,
              size: layer.size || size,
              intensity: layer.intensity || intensity,
              position: layer.position || position,
            })
            
            const layerStyle = {
              ...animationStyle,
              ...(layer.offset && {
                transform: `translate(${layer.offset.x}px, ${layer.offset.y}px)`,
              }),
            }

            return (
              <div
                key={index}
                className={layerClasses}
                style={layerStyle}
              />
            )
          })}
        </div>
      )
    }

    // Single glow element
    return (
      <div
        ref={ref}
        className={cn(baseClasses, animated && "animate-pulse", className)}
        style={animationStyle}
        {...props}
      />
    )
  }
)

Glow.displayName = "Glow"

export { Glow, glowVariants }
